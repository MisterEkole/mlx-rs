fn main() {
    
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        return;
    }

    cc::Build::new()
        .file("src/bridge.m")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        // Do NOT pass -framework here; frameworks are linked by rustc, not clang.
        // The .compile() call produces a static lib; the cargo:rustc-link-lib lines
        // below tell rustc's linker to pull in the frameworks.
        .compile("ane_bridge");

    // Framework linking — these go to rustc's linker, not to clang
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=framework=CoreML");

    println!("cargo:rerun-if-changed=src/bridge.m");
    println!("cargo:rerun-if-changed=src/lib.rs");
}
