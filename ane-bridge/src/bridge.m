// ane-bridge/src/bridge.m
//
// Obj-C bridge that compiles linear layers for the Apple Neural Engine.
//
// Based on reverse engineering by Manjeet Singh (maderix):
//   https://github.com/maderix/ANE
//
// Compile pipeline per layer:
//   1. Build a CoreML NeuralNetwork binary protobuf (innerProduct, fp16 weights).
//   2. Write to a temp .mlmodel file and compile via MLModel.compileModelAtURL:.
//      CoreML routes innerProduct to ANE on Apple Silicon.
//   3. Load via MLModel.modelWithContentsOfURL: (CoreML fallback path).
//   4. Load into the ANE daemon via _ANEClient.loadModel: (fast path).
//   5. Create fp16 IOSurfaces, build _ANERequest, call mapIOSurfacesWithModel:.
//
// Execute per call:
//   Fast path  (use_fast_path=1): write IOSurface → evaluateWithModel: → read IOSurface
//   Fallback   (use_fast_path=0): MLMultiArray → predictionFromFeatures:

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ── IOSurface compatibility shim ───────────────────────────────────────────
//
// In macOS 26, IOSurface objects no longer respond to -ioSurface.
// mapIOSurfacesWithModel: calls -ioSurface on each surface to get the
// IOSurfaceRef for DMA mapping. Without this selector it throws an
// unrecognized-selector exception.
//
// Fix: inject -ioSurface via class_addMethod, returning self.
// IOSurfaceRef is toll-free bridged to IOSurface, so returning self is correct.
//
// class_addMethod is used instead of an @interface category because the
// -fmodules flag makes the IOSurface ObjC class invisible to category syntax.

// Add a missing selector to IOSurface (object return type).
static void add_ios_id_method(Class cls, const char *name, id (^block)(id)) {
    SEL sel = sel_registerName(name);
    if (class_getInstanceMethod(cls, sel)) return; // already present — no-op
    IMP imp = imp_implementationWithBlock(block);
    BOOL ok  = class_addMethod(cls, sel, imp, "@@:");
    NSLog(@"[ane_bridge] -[IOSurface %s] shim: %s", name, ok ? "ok" : "failed");
}

// Add a missing selector to IOSurface (NSUInteger return type).
static void add_ios_uint_method(Class cls, const char *name, NSUInteger (^block)(id)) {
    SEL sel = sel_registerName(name);
    if (class_getInstanceMethod(cls, sel)) return;
    IMP imp = imp_implementationWithBlock(block);
    BOOL ok  = class_addMethod(cls, sel, imp, "Q@:");
    NSLog(@"[ane_bridge] -[IOSurface %s] shim: %s", name, ok ? "ok" : "failed");
}

static void install_iosurface_compat(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        Class cls = NSClassFromString(@"IOSurface");
        if (!cls) { NSLog(@"[ane_bridge] IOSurface class not found"); return; }

        // -ioSurface: the ANE framework calls this to get the IOSurface from a
        //   descriptor object. IOSurfaceRef IS the IOSurface (toll-free bridge),
        //   so returning self is correct.
        add_ios_id_method(cls, "ioSurface", ^id(id self_) { return self_; });

        // -startOffset: byte offset into the IOSurface backing allocation where
        //   the tensor data begins. For a full-surface we own, this is always 0.
        add_ios_uint_method(cls, "startOffset", ^NSUInteger(id self_) { return 0; });
    });
}

// ── Compile counter & budget ───────────────────────────────────────────────

static int g_compile_cnt   = 0;
static int g_fast_path_cnt = 0;  // active mapIOSurfaces(YES) slots in ANE daemon

#define ANE_COMPILE_LIMIT   110
// The ANE daemon's inference-cache slot pool holds 24 active mappings per
// sharedConnection. Beyond this, mapIOSurfaces(YES) returns Code=13/0x1D.
// We cap fast-path slots here and silently use CoreML for the remainder.
#define ANE_FAST_PATH_LIMIT  24

// ── ANE availability ───────────────────────────────────────────────────────
//
// We check that CoreML is available and that the device supports ANE
// (i.e., Apple Silicon — MLComputeUnitsCPUAndNeuralEngine succeeds at load).

int ane_is_available(void) {
    // CoreML is always present on macOS 10.13+. On Apple Silicon the ANE is
    // always present. We return 1 unconditionally on macOS — the actual test
    // happens at compile time when MLModel.compileModelAtURL: is called.
    return 1;
}

int ane_compile_count(void) {
    return g_compile_cnt;
}

// ── fp16 ↔ float32 helpers ─────────────────────────────────────────────────

static inline float fp16_to_f32(uint16_t h) {
    __fp16 tmp;
    memcpy(&tmp, &h, 2);
    return (float)tmp;
}

static inline uint16_t f32_to_fp16(float f) {
    __fp16 tmp = (__fp16)f;
    uint16_t h;
    memcpy(&h, &tmp, 2);
    return h;
}

// ── Minimal protobuf encoder ───────────────────────────────────────────────
//
// Supports just enough to write a CoreML NeuralNetwork binary protobuf.
// Wire types used: 0=varint, 2=length-delimited.

typedef struct {
    uint8_t *buf;
    size_t   pos;
    size_t   cap;
    int      oom;
} PB;

static void pb_init(PB *pb, size_t initial_cap) {
    pb->buf = (uint8_t *)malloc(initial_cap);
    pb->pos = 0;
    pb->cap = pb->buf ? initial_cap : 0;
    pb->oom = !pb->buf;
}

static void pb_free(PB *pb) { free(pb->buf); pb->buf = NULL; }

static void pb_ensure(PB *pb, size_t extra) {
    if (pb->oom) return;
    if (pb->pos + extra <= pb->cap) return;
    size_t new_cap = pb->cap * 2 + extra + 64;
    uint8_t *nb = (uint8_t *)realloc(pb->buf, new_cap);
    if (!nb) { pb->oom = 1; return; }
    pb->buf = nb;
    pb->cap = new_cap;
}

static void pb_varint(PB *pb, uint64_t v) {
    pb_ensure(pb, 10);
    if (pb->oom) return;
    do {
        uint8_t b = (uint8_t)(v & 0x7F);
        v >>= 7;
        pb->buf[pb->pos++] = b | (v ? 0x80 : 0);
    } while (v);
}

static void pb_tag(PB *pb, int field, int wire) {
    pb_varint(pb, ((uint64_t)field << 3) | (unsigned)wire);
}

// field N, wire 0 (varint value)
static void pb_varint_field(PB *pb, int field, uint64_t v) {
    pb_tag(pb, field, 0);
    pb_varint(pb, v);
}

// field N, wire 2 (length-delimited: bytes/string/embedded message)
static void pb_bytes_field(PB *pb, int field, const uint8_t *data, size_t len) {
    pb_tag(pb, field, 2);
    pb_varint(pb, (uint64_t)len);
    pb_ensure(pb, len);
    if (!pb->oom) { memcpy(pb->buf + pb->pos, data, len); pb->pos += len; }
}

static void pb_string_field(PB *pb, int field, const char *str) {
    pb_bytes_field(pb, field, (const uint8_t *)str, strlen(str));
}

// ── CoreML protobuf builder ────────────────────────────────────────────────
//
// Verified field numbers (from coremltools Python proto definitions):
//
// Model          : specificationVersion=1, description=2, neuralNetwork=500
// ModelDescription: input=1, output=10
// FeatureDescription: name=1, type=3
// FeatureType    : multiArrayType=5
// ArrayFeatureType: shape=1 (repeated int64), dataType=2  FLOAT32=65568
// NeuralNetwork  : layers=1, arrayInputShapeMapping=5
//   EXACT_ARRAY_MAPPING=1 (required for specVersion>=4)
// NeuralNetworkLayer: name=1, input=2, output=3,
//   inputTensor=4, outputTensor=5, innerProduct=140
// NeuralNetworkLayer.Tensor: rank=1, dimValue=2
// InnerProductLayerParams: inputChannels=1, outputChannels=2,
//   hasBias=10, weights=20, bias=21
// WeightParams   : floatValue=1, float16Value=2

// ArrayFeatureType for a 2-D array [batch × features] with float16 elements.
// We use fp16 so the input/output can be memcpy'd directly from/to our fp16
// buffers — no fp16↔f32 conversion needed at inference time.
// We use 2D I/O so the whole batch can be processed in ONE prediction call,
// avoiding the per-sample MLMultiArray overhead.
static NSData *build_array_feature_type_2d(int64_t batch, int64_t features) {
    PB pb; pb_init(&pb, 48);
    // field 1 (shape): repeated int64 — two elements (batch, features)
    pb_varint_field(&pb, 1, (uint64_t)batch);
    pb_varint_field(&pb, 1, (uint64_t)features);
    // field 2 (dataType): FLOAT16 = 65552
    pb_varint_field(&pb, 2, 65552);
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// FeatureType wrapping a 2D ArrayFeatureType
static NSData *build_feature_type_2d(int64_t batch, int64_t features) {
    NSData *aft = build_array_feature_type_2d(batch, features);
    PB pb; pb_init(&pb, aft.length + 8);
    // field 5 (multiArrayType)
    pb_bytes_field(&pb, 5, (const uint8_t *)aft.bytes, aft.length);
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// FeatureDescription: name + 2D type [batch × features]
static NSData *build_feature_desc_2d(const char *name, int64_t batch, int64_t features) {
    NSData *ft = build_feature_type_2d(batch, features);
    PB pb; pb_init(&pb, 64 + ft.length);
    // field 1 (name)
    pb_string_field(&pb, 1, name);
    // field 3 (type)
    pb_bytes_field(&pb, 3, (const uint8_t *)ft.bytes, ft.length);
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// ModelDescription: input [batch × in_f], output [batch × out_f]
static NSData *build_model_description(int batch, int in_f, int out_f) {
    NSData *inp  = build_feature_desc_2d("input",  (int64_t)batch, (int64_t)in_f);
    NSData *outp = build_feature_desc_2d("output", (int64_t)batch, (int64_t)out_f);
    PB pb; pb_init(&pb, 32 + inp.length + outp.length);
    // field 1 (input)
    pb_bytes_field(&pb, 1, (const uint8_t *)inp.bytes,  inp.length);
    // field 10 (output) — CoreML ModelDescription.output is field 10, not 2
    pb_bytes_field(&pb, 10, (const uint8_t *)outp.bytes, outp.length);
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// WeightParams: fp16 bytes stored in float16Value
// Verified: WeightParams.float16Value = field 2 (not 6)
static NSData *build_weight_params_fp16(const uint16_t *fp16_data, int count) {
    PB pb; pb_init(&pb, (size_t)count * 2 + 16);
    // field 2 (float16Value): raw bytes
    pb_bytes_field(&pb, 2, (const uint8_t *)fp16_data, (size_t)count * 2);
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// InnerProductLayerParams
// Verified field numbers:
//   inputChannels = 1, outputChannels = 2, hasBias = 10, weights = 20, bias = 21
static NSData *build_inner_product(int in_f, int out_f,
                                    const uint16_t *weights,
                                    const uint16_t *bias) {
    NSData *wp = build_weight_params_fp16(weights, in_f * out_f);
    NSData *bp = bias ? build_weight_params_fp16(bias, out_f) : nil;

    PB pb; pb_init(&pb, 32 + wp.length + (bp ? bp.length : 0));
    // field 1 (inputChannels)
    pb_varint_field(&pb, 1, (uint64_t)in_f);
    // field 2 (outputChannels)
    pb_varint_field(&pb, 2, (uint64_t)out_f);
    // field 10 (hasBias)
    pb_varint_field(&pb, 10, bias ? 1 : 0);
    // field 20 (weights)
    pb_bytes_field(&pb, 20, (const uint8_t *)wp.bytes, wp.length);
    // field 21 (bias)
    if (bp) pb_bytes_field(&pb, 21, (const uint8_t *)bp.bytes, bp.length);

    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// Tensor: rank=2 + two dimValues for EXACT_ARRAY_MAPPING shape declarations
// NeuralNetworkLayer.Tensor: rank=1, dimValue=2
static NSData *build_tensor_2d(int64_t batch, int64_t dim) {
    PB pb; pb_init(&pb, 32);
    pb_varint_field(&pb, 1, 2);              // rank = 2
    pb_varint_field(&pb, 2, (uint64_t)batch); // dimValue[0] = batch
    pb_varint_field(&pb, 2, (uint64_t)dim);   // dimValue[1] = dim
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// NeuralNetworkLayer wrapping an InnerProductLayerParams
//
// Input/output tensors are declared as 2D [batch × features] so the entire
// batch can be submitted as ONE MLMultiArray in a single prediction call.
//
// Verified field numbers (from coremltools proto):
//   name=1, input=2, output=3, inputTensor=4, outputTensor=5, innerProduct=140
static NSData *build_nn_layer(int batch, int in_f, int out_f,
                               const uint16_t *weights,
                               const uint16_t *bias) {
    NSData *ip       = build_inner_product(in_f, out_f, weights, bias);
    NSData *in_tens  = build_tensor_2d((int64_t)batch, (int64_t)in_f);
    NSData *out_tens = build_tensor_2d((int64_t)batch, (int64_t)out_f);

    PB pb; pb_init(&pb, 128 + ip.length);
    // field 1 (name)
    pb_string_field(&pb, 1, "linear");
    // field 2 (input): string feature name
    pb_string_field(&pb, 2, "input");
    // field 3 (output): string feature name
    pb_string_field(&pb, 3, "output");
    // field 4 (inputTensor): 2D shape [batch, in_features]
    pb_bytes_field(&pb, 4, (const uint8_t *)in_tens.bytes,  in_tens.length);
    // field 5 (outputTensor): 2D shape [batch, out_features]
    pb_bytes_field(&pb, 5, (const uint8_t *)out_tens.bytes, out_tens.length);
    // field 140 (innerProduct): InnerProductLayerParams
    pb_bytes_field(&pb, 140, (const uint8_t *)ip.bytes, ip.length);
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// NeuralNetwork: one layer, EXACT_ARRAY_MAPPING (required for specVersion >= 4)
// NeuralNetwork.arrayInputShapeMapping = 5, EXACT_ARRAY_MAPPING = 1
static NSData *build_neural_network(int batch, int in_f, int out_f,
                                     const uint16_t *weights,
                                     const uint16_t *bias) {
    NSData *layer = build_nn_layer(batch, in_f, out_f, weights, bias);
    PB pb; pb_init(&pb, layer.length + 16);
    // field 1 (layers)
    pb_bytes_field(&pb, 1, (const uint8_t *)layer.bytes, layer.length);
    // field 5 (arrayInputShapeMapping) = 1 (EXACT_ARRAY_MAPPING)
    pb_varint_field(&pb, 5, 1);
    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// Top-level CoreML Model: specVersion=4, description, neuralNetwork
//
// Verified field numbers from coremltools proto definitions:
//   Model.specificationVersion = 1
//   Model.description          = 2  (ModelDescription)
//   Model.neuralNetwork        = 500 (NeuralNetwork)
//   ModelDescription.input     = 1  (repeated FeatureDescription)
//   ModelDescription.output    = 10 (repeated FeatureDescription)
//   FeatureDescription.name    = 1
//   FeatureDescription.type    = 3  (FeatureType)
//   FeatureType.multiArrayType = 5  (ArrayFeatureType)
//   ArrayFeatureType.shape     = 1  (repeated int64)
//   ArrayFeatureType.dataType  = 2  (FLOAT32=65568)
//   NeuralNetwork.layers       = 1  (repeated NeuralNetworkLayer)
//   NeuralNetworkLayer.name    = 1, .input = 2, .output = 3, .innerProduct = 5
//   InnerProductLayerParams:   inputChannels=1, outputChannels=2, hasBias=3,
//                              weights=4, bias=5
//   WeightParams.float16Value  = 6  (bytes)
static NSData *build_coreml_proto(int batch, int in_f, int out_f,
                                   const uint16_t *weights,
                                   const uint16_t *bias) {
    NSData *desc = build_model_description(batch, in_f, out_f);
    NSData *nn   = build_neural_network(batch, in_f, out_f, weights, bias);

    PB pb; pb_init(&pb, 32 + desc.length + nn.length);
    // field 1 (specificationVersion) = 7 (required for FLOAT16 feature types)
    pb_varint_field(&pb, 1, 7);
    // field 2 (description)
    pb_bytes_field(&pb, 2, (const uint8_t *)desc.bytes, desc.length);
    // field 500 (neuralNetwork) — tag varint = (500 << 3) | 2 = 4002
    pb_bytes_field(&pb, 500, (const uint8_t *)nn.bytes, nn.length);

    NSData *r = [NSData dataWithBytes:pb.buf length:pb.pos];
    pb_free(&pb);
    return r;
}

// ── AneProgram opaque handle ───────────────────────────────────────────────
//
// Contains both a CoreML fallback path (always set) and an optional fast
// _ANEClient IOSurface path (set when the private API is available).
//
// Fast path layout (from model.espresso.shape inspection):
//   Input IOSurface:  Width=in_features,  Height=batch_size, 2 bytes/element
//   Output IOSurface: Width=out_features, Height=batch_size, 2 bytes/element
// Both match our row-major [batch, features] layout exactly — direct memcpy.

typedef struct AneProgram {
    // CoreML fallback (always present)
    void        *mlmodel;       // MLModel*

    // Private _ANEClient fast path (NULL if setup failed)
    void        *ane_model;     // _ANEModel* (private, retained via CFBridgingRetain)
    void        *ane_request;   // _ANERequest* (private, retained via CFBridgingRetain)
    IOSurfaceRef input_ios;     // input IOSurface  (Width=in_f,  Height=batch, fp16)
    IOSurfaceRef output_ios;    // output IOSurface (Width=out_f, Height=batch, fp16)
    size_t       input_bpr;     // bytes per row for input IOSurface
    size_t       output_bpr;    // bytes per row for output IOSurface
    int          use_fast_path; // 1 when IOSurface/_ANEClient path is active

    int  in_features;
    int  out_features;
    int  batch_size;
} AneProgram;

// ── IOSurface helpers ──────────────────────────────────────────────────────

// Align n up to the nearest multiple of align (must be power of 2).
static inline size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

// Create an IOSurface via C API with Width=w, Height=h, 2 bytes/element (fp16).
// bytesPerRow is aligned to 16 bytes (IOSurface minimum requirement).
// Returns a +1 CF-retained IOSurfaceRef, or NULL on failure.
static IOSurfaceRef create_fp16_ios(int w, int h, size_t *bpr_out) {
    size_t bpr = align_up((size_t)w * 2, 16);
    NSDictionary *props = @{
        (__bridge NSString*)kIOSurfaceWidth:           @(w),
        (__bridge NSString*)kIOSurfaceHeight:          @(h),
        (__bridge NSString*)kIOSurfaceBytesPerElement: @2,
        (__bridge NSString*)kIOSurfaceBytesPerRow:     @(bpr),
        (__bridge NSString*)kIOSurfaceAllocSize:       @((size_t)h * bpr),
    };
    IOSurfaceRef ref = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    if (ref && bpr_out) *bpr_out = bpr;
    return ref;  // +1 from IOSurfaceCreate (CF_RETURNS_RETAINED)
}

// Copy a contiguous fp16 matrix [rows × cols] into an IOSurface.
// When bytesPerRow == cols*2 (no padding) this is a single memcpy.
static void write_surface(IOSurfaceRef ios, const uint16_t *src,
                          int rows, int cols, size_t bpr) {
    IOSurfaceLock(ios, 0, NULL);
    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(ios);
    size_t row_bytes = (size_t)cols * 2;
    if (bpr == row_bytes) {
        memcpy(base, src, (size_t)rows * row_bytes);
    } else {
        for (int r = 0; r < rows; r++)
            memcpy(base + (size_t)r * bpr, src + (size_t)r * cols, row_bytes);
    }
    IOSurfaceUnlock(ios, 0, NULL);
}

// Copy from an IOSurface into a contiguous fp16 matrix.
static void read_surface(uint16_t *dst, IOSurfaceRef ios,
                         int rows, int cols, size_t bpr) {
    IOSurfaceLock(ios, kIOSurfaceLockReadOnly, NULL);
    const uint8_t *base = (const uint8_t *)IOSurfaceGetBaseAddress(ios);
    size_t row_bytes = (size_t)cols * 2;
    if (bpr == row_bytes) {
        memcpy(dst, base, (size_t)rows * row_bytes);
    } else {
        for (int r = 0; r < rows; r++)
            memcpy(dst + (size_t)r * cols, base + (size_t)r * bpr, row_bytes);
    }
    IOSurfaceUnlock(ios, kIOSurfaceLockReadOnly, NULL);
}

// ── ane_linear_compile ─────────────────────────────────────────────────────
//
// Encodes a CoreML NeuralNetwork binary protobuf, writes it to a temp .mlmodel,
// and compiles via [MLModel compileModelAtURL:] — CoreML routes innerProduct
// to the ANE. Loads via [MLModel modelWithContentsOfURL:] for the CoreML
// fallback path, then attempts to wire up the fast _ANEClient IOSurface path.
// Any failure in the fast-path setup breaks out of the do{}while(0) block and
// leaves the CoreML fallback in place — no error is logged.

AneProgram *ane_linear_compile(const uint16_t *weights,
                                const uint16_t *bias,
                                int in_features,
                                int out_features,
                                int batch_size) {
    if (g_compile_cnt >= ANE_COMPILE_LIMIT) return NULL;

    // Inject -ioSurface on first call so mapIOSurfacesWithModel: doesn't throw.
    install_iosurface_compat();

    // Build CoreML binary protobuf
    NSData *model_data = build_coreml_proto(batch_size, in_features, out_features,
                                             weights, bias);
    if (!model_data || model_data.length == 0) {
        NSLog(@"[ane_bridge] failed to build CoreML proto (OOM?)");
        return NULL;
    }

    // Write protobuf to a temp file and compile via MLModel
    NSString *fname = [NSString stringWithFormat:@"ane_linear_%d_%d_%d.mlmodel",
                       in_features, out_features, g_compile_cnt];
    NSURL *src_url = [NSURL fileURLWithPath:
                      [NSTemporaryDirectory() stringByAppendingPathComponent:fname]];

    NSError *ns_err = nil;
    if (![model_data writeToURL:src_url options:NSDataWritingAtomic error:&ns_err]) {
        NSLog(@"[ane_bridge] failed to write temp .mlmodel: %@", ns_err);
        return NULL;
    }

    ns_err = nil;
    NSURL *compiled_url = [MLModel compileModelAtURL:src_url error:&ns_err];
    [[NSFileManager defaultManager] removeItemAtURL:src_url error:nil];

    if (!compiled_url || ns_err) {
        NSLog(@"[ane_bridge] MLModel.compileModelAtURL failed: %@", ns_err);
        return NULL;
    }

    // Load CoreML model — always present as the fallback execute path
    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = 3;  // MLComputeUnitsCPUAndNeuralEngine

    ns_err = nil;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled_url
                                       configuration:config
                                               error:&ns_err];
    if (!model || ns_err) {
        NSLog(@"[ane_bridge] MLModel load failed: %@", ns_err);
        [[NSFileManager defaultManager] removeItemAtURL:compiled_url error:nil];
        return NULL;
    }

    g_compile_cnt++;

    AneProgram *prog = (AneProgram *)calloc(1, sizeof(AneProgram));
    if (!prog) {
        [[NSFileManager defaultManager] removeItemAtURL:compiled_url error:nil];
        return NULL;
    }
    prog->mlmodel      = (void *)CFBridgingRetain(model);
    prog->in_features  = in_features;
    prog->out_features = out_features;
    prog->batch_size   = batch_size;

    // Attempt fast _ANEClient IOSurface path. Wrapped in do{}while(0) so any
    // failure breaks out cleanly, leaving the CoreML fallback path in place.
    do {
        // Hard cap: the ANE daemon's inference-cache slot pool is limited to
        // ANE_FAST_PATH_LIMIT active mapIOSurfaces(YES) mappings per connection.
        // Once the cap is reached, fall through to CoreML silently — no error log.
        if (g_fast_path_cnt >= ANE_FAST_PATH_LIMIT) {
            break;
        }

        // Load AppleNeuralEngine.framework (no-op if already loaded by CoreML)
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
               RTLD_NOW | RTLD_GLOBAL);

        Class ANEModelCls   = NSClassFromString(@"_ANEModel");
        Class ANEClientCls  = NSClassFromString(@"_ANEClient");
        Class ANERequestCls = NSClassFromString(@"_ANERequest");
        if (!ANEModelCls || !ANEClientCls || !ANERequestCls) {
            NSLog(@"[ane_bridge] private ANE classes not found — using CoreML fallback");
            break;
        }

        // Load _ANEModel from compiled bundle
        typedef id (*MsgId_Url_Id)(Class, SEL, NSURL *, id);
        id ane_model = ((MsgId_Url_Id)objc_msgSend)(
            ANEModelCls,
            sel_registerName("modelAtURL:key:"),
            compiled_url, nil);
        if (!ane_model) {
            NSLog(@"[ane_bridge] _ANEModel modelAtURL:key: returned nil");
            break;
        }

        // Get shared client and load model into the ANE daemon
        typedef id (*MsgId)(Class, SEL);
        id client = ((MsgId)objc_msgSend)(ANEClientCls,
                                          sel_registerName("sharedConnection"));
        if (!client) {
            NSLog(@"[ane_bridge] _ANEClient sharedConnection returned nil");
            break;
        }

        typedef NSInteger (*MsgNSInt_Id_Id_Int_ErrPtr)(id, SEL, id, id, NSInteger, NSError **);
        ns_err = nil;
        NSInteger rc = ((MsgNSInt_Id_Id_Int_ErrPtr)objc_msgSend)(
            client,
            sel_registerName("loadModel:options:qos:error:"),
            ane_model, @{}, (NSInteger)25 /* NSQualityOfServiceUserInitiated */, &ns_err);
        // rc=0 or rc=1 with nil error both indicate success.
        // Only break on rc<0 or (rc!=0 and ns_err is set).
        if (ns_err || rc < 0) {
            NSLog(@"[ane_bridge] loadModel:options:qos:error: failed rc=%ld %@",
                  (long)rc, ns_err);
            break;
        }
        NSLog(@"[ane_bridge] loadModel: rc=%ld (success)", (long)rc);

        // Create fp16 IOSurfaces: Width=features, Height=batch.
        // Matches the row-major [batch, features] layout directly — no transpose needed.
        size_t inp_bpr = 0, out_bpr = 0;
        IOSurfaceRef inp_ios = create_fp16_ios(in_features,  batch_size, &inp_bpr);
        IOSurfaceRef out_ios = create_fp16_ios(out_features, batch_size, &out_bpr);
        if (!inp_ios || !out_ios) {
            NSLog(@"[ane_bridge] IOSurface creation failed");
            if (inp_ios) CFRelease(inp_ios);
            if (out_ios) CFRelease(out_ios);
            break;
        }

        // Query symbol indices (populated after loadModel:).
        // Fall back to @[@0] if empty — index 0 is correct for single-input models.
        typedef id (*MsgId_Int)(id, SEL, int);
        NSIndexSet *inIdx  = ((MsgId_Int)objc_msgSend)(
            ane_model, sel_registerName("inputSymbolIndicesForProcedureIndex:"), 0);
        NSIndexSet *outIdx = ((MsgId_Int)objc_msgSend)(
            ane_model, sel_registerName("outputSymbolIndicesForProcedureIndex:"), 0);
        NSArray *inp_indices = (inIdx  && inIdx.count  > 0) ? @[@(inIdx.firstIndex)]  : @[@0];
        NSArray *out_indices = (outIdx && outIdx.count > 0) ? @[@(outIdx.firstIndex)] : @[@0];

        // Build _ANERequest with bridge-cast IOSurfaceRefs.
        // _ANERequest retains the surfaces for the lifetime of the request.
        NSArray *inp_surfs = @[(__bridge id)inp_ios];
        NSArray *out_surfs = @[(__bridge id)out_ios];
        typedef id (*MsgReqCreate)(Class, SEL, NSArray*, NSArray*, NSArray*, NSArray*, int);
        id request = ((MsgReqCreate)objc_msgSend)(
            ANERequestCls,
            sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
            inp_surfs, inp_indices, out_surfs, out_indices, 0);
        if (!request) {
            NSLog(@"[ane_bridge] _ANERequest creation failed");
            CFRelease(inp_ios);
            CFRelease(out_ios);
            break;
        }

        // Pre-map IOSurfaces: wires DMA access in the ANE daemon for all future calls.
        // cacheInference:YES caches the mapping — required for fast repeated evaluations.
        // Without pre-mapping, evaluateWithModel: returns Code=3.
        // mapIOSurfaces is capped by ANE_FAST_PATH_LIMIT above, so Code=13 is avoided.
        typedef BOOL (*MsgMap)(id, SEL, id, id, BOOL, NSError **);
        NSError *map_err = nil;
        BOOL mapped = NO;
        @try {
            mapped = ((MsgMap)objc_msgSend)(
                client,
                sel_registerName("mapIOSurfacesWithModel:request:cacheInference:error:"),
                ane_model, request, YES, &map_err);
        } @catch (NSException *ex) {
            NSLog(@"[ane_bridge] mapIOSurfaces threw: %@ — %@", ex.name, ex.reason);
            CFRelease(inp_ios);
            CFRelease(out_ios);
            break;
        }
        if (!mapped || map_err) {
            NSLog(@"[ane_bridge] mapIOSurfaces failed: %@", map_err);
            CFRelease(inp_ios);
            CFRelease(out_ios);
            break;
        }

        // Success — fast path is live.
        // inp_ios / out_ios have +1 retain from IOSurfaceCreate; transferred to prog.
        prog->ane_model     = (void *)CFBridgingRetain(ane_model);
        prog->ane_request   = (void *)CFBridgingRetain(request);
        prog->input_ios     = inp_ios;
        prog->output_ios    = out_ios;
        prog->input_bpr     = inp_bpr;
        prog->output_bpr    = out_bpr;
        prog->use_fast_path = 1;
        g_fast_path_cnt++;
        NSLog(@"[ane_bridge] fast _ANEClient path active for %dx%d batch=%d",
              in_features, out_features, batch_size);
    } while (0);

    // Compiled bundle can be deleted now — both MLModel and _ANEModel have
    // loaded what they need from it (the latter copies to the ANE daemon).
    [[NSFileManager defaultManager] removeItemAtURL:compiled_url error:nil];

    return prog;
}

// ── ane_linear_execute ─────────────────────────────────────────────────────
//
// Fast path  (use_fast_path == 1):
//   Write fp16 input to IOSurface, call evaluateWithModel: (bypasses the
//   CoreML stack), read fp16 output from IOSurface.
//
// Fallback (use_fast_path == 0 or fast path errors):
//   Single predictionFromFeatures: call via CoreML.
//
// Input/output are fp16 row-major [batch_size × features].

static int coreml_execute(AneProgram *prog,
                           const uint16_t *input_fp16,
                           uint16_t *output_fp16,
                           int batch_size) {
    MLModel *model = (__bridge MLModel *)prog->mlmodel;
    int in_f  = prog->in_features;
    int out_f = prog->out_features;

    NSError *ns_err = nil;
    NSArray<NSNumber*> *inp_shape   = @[@(batch_size), @(in_f)];
    NSArray<NSNumber*> *inp_strides = @[@(in_f), @(1)];
    MLMultiArray *input_arr = [[MLMultiArray alloc]
        initWithDataPointer:(void *)input_fp16
                     shape:inp_shape
                  dataType:MLMultiArrayDataTypeFloat16
                   strides:inp_strides
               deallocator:nil
                     error:&ns_err];
    if (!input_arr) return -2;

    MLFeatureValue *fv = [MLFeatureValue featureValueWithMultiArray:input_arr];
    MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"input": fv} error:&ns_err];
    if (!fp) return -3;

    id<MLFeatureProvider> result = [model predictionFromFeatures:fp error:&ns_err];
    if (!result || ns_err) return -4;

    MLMultiArray *out_arr = [[result featureValueForName:@"output"] multiArrayValue];
    if (!out_arr) return -5;

    memcpy(output_fp16, out_arr.dataPointer, (size_t)(batch_size * out_f) * 2);
    return 0;
}

int ane_linear_execute(AneProgram *prog,
                       const uint16_t *input_fp16,
                       uint16_t *output_fp16,
                       int batch_size) {
    if (!prog || !input_fp16 || !output_fp16) return -1;

    int in_f  = prog->in_features;
    int out_f = prog->out_features;
    (void)out_f;  // used below via read_surface; suppress unused warning if fallback taken

    // Fast path: _ANEClient + pre-mapped IOSurfaces────
    if (prog->use_fast_path) {
        Class ANEClientCls = NSClassFromString(@"_ANEClient");
        typedef id (*MsgId)(Class, SEL);
        id client = ((MsgId)objc_msgSend)(ANEClientCls,
                                          sel_registerName("sharedConnection"));
        if (!client) goto fallback;

        // Write input into IOSurface (lock → memcpy → unlock)
        write_surface(prog->input_ios, input_fp16,
                      batch_size, in_f, prog->input_bpr);

        // Execute on ANE — IOSurfaces were pre-mapped at compile time (YES).
        // evaluateWithModel uses the cached mapping directly: no re-mapping overhead.
        {
        typedef NSInteger (*MsgEval)(id, SEL, id, id, id, NSInteger, NSError **);
        NSError *ns_err = nil;
        NSInteger rc = -1;
        @try {
            rc = ((MsgEval)objc_msgSend)(
                client,
                sel_registerName("evaluateWithModel:options:request:qos:error:"),
                (__bridge id)(CFTypeRef)prog->ane_model,
                @{},
                (__bridge id)(CFTypeRef)prog->ane_request,
                (NSInteger)25 /* NSQualityOfServiceUserInitiated */,
                &ns_err);
        } @catch (NSException *ex) {
            NSLog(@"[ane_bridge] evaluateWithModel threw: %@ %@", ex.name, ex.reason);
            goto fallback;
        }
        if (ns_err || rc < 0) {
            NSLog(@"[ane_bridge] evaluateWithModel failed rc=%ld %@", (long)rc, ns_err);
            goto fallback;
        }
        }

        // Read output from pre-mapped IOSurface (lock → memcpy → unlock)
        read_surface(output_fp16, prog->output_ios,
                     batch_size, out_f, prog->output_bpr);
        return 0;
    }

fallback:
    return coreml_execute(prog, input_fp16, output_fp16, batch_size);
}

// ── ane_program_free ───────────────────────────────────────────────────────

void ane_program_free(AneProgram *prog) {
    if (!prog) return;
    if (prog->mlmodel)    CFRelease((CFTypeRef)prog->mlmodel);
    if (prog->ane_model)  CFRelease((CFTypeRef)prog->ane_model);
    if (prog->ane_request)CFRelease((CFTypeRef)prog->ane_request);
    if (prog->input_ios)  CFRelease((CFTypeRef)prog->input_ios);
    if (prog->output_ios) CFRelease((CFTypeRef)prog->output_ios);
    free(prog);
}

// ── ane_diagnose ───────────────────────────────────────────────────────────
//
// Prints runtime info and ANE private class method lists for debugging.

static void dump_methods(const char *class_name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:class_name]);
    if (!cls) {
        printf("  [ane_diagnose] '%s' — NOT FOUND\n", class_name);
        return;
    }
    printf("  [ane_diagnose] '%s' found\n", class_name);

    unsigned int n = 0;
    Method *im = class_copyMethodList(cls, &n);
    printf("  [ane_diagnose]   %u instance methods:\n", n);
    for (unsigned int i = 0; i < n; i++)
        printf("  [ane_diagnose]     - %s\n", sel_getName(method_getName(im[i])));
    if (im) free(im);

    Method *cm = class_copyMethodList(object_getClass(cls), &n);
    printf("  [ane_diagnose]   %u class methods:\n", n);
    for (unsigned int i = 0; i < n; i++)
        printf("  [ane_diagnose]     + %s\n", sel_getName(method_getName(cm[i])));
    if (cm) free(cm);
}

void ane_diagnose(void) {
    printf("\n[ane_diagnose] === ANE bridge runtime info ===\n");
    printf("[ane_diagnose] Using public CoreML API (MLModel)\n");
    printf("[ane_diagnose] compile_count = %d / %d\n", g_compile_cnt, ANE_COMPILE_LIMIT);

    // Verify MLModelConfiguration is usable
    @try {
        MLModelConfiguration *cfg = [MLModelConfiguration new];
        cfg.computeUnits = 3;  // MLComputeUnitsCPUAndNeuralEngine
        printf("[ane_diagnose] MLModelConfiguration OK, computeUnits=3 (CPU+NE)\n");
    } @catch (NSException *ex) {
        printf("[ane_diagnose] MLModelConfiguration threw: %s\n", ex.reason.UTF8String);
    }

    // Load ANE framework and enumerate all classes
    const char *ane_path =
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
    void *ane_handle = dlopen(ane_path, RTLD_NOW | RTLD_GLOBAL);
    if (ane_handle) {
        printf("[ane_diagnose] AppleNeuralEngine.framework loaded\n\n");

        // Enumerate all classes in the framework
        unsigned int cls_count = 0;
        const char **cls_names = objc_copyClassNamesForImage(ane_path, &cls_count);
        printf("[ane_diagnose] %u classes in AppleNeuralEngine.framework:\n", cls_count);
        for (unsigned int i = 0; i < cls_count; i++)
            printf("[ane_diagnose]   %s\n", cls_names[i]);
        if (cls_names) free((void *)cls_names);
        printf("\n");

        // Dump methods on the private classes used by the fast path
        dump_methods("_ANEModel");        printf("\n");
        dump_methods("_ANEClient");       printf("\n");
        dump_methods("_ANERequest");      printf("\n");
        dump_methods("_ANEIOSurface");    printf("\n");
        dump_methods("_ANEDeviceInfo");   printf("\n");
    } else {
        printf("[ane_diagnose] AppleNeuralEngine.framework not loaded: %s\n", dlerror());
    }

    printf("[ane_diagnose] ===========================\n\n");
}
