use std::env;
use std::path::PathBuf;

fn main() {
    let mlx_c_path = env::var("MLX_C_PATH").unwrap_or_else(|_| {
        let fallback = "/Users/ekole/Dev/mlx-c";
        
        println!("cargo:warning=MLX_C_PATH is not set. Falling back to: {}", fallback);
        fallback.to_string()
    });
    
    let manifest_dir = PathBuf::from(mlx_c_path);

    // Link the C-Wrapper (mlxc)
    println!("cargo:rustc-link-search=native={}/build", manifest_dir.display());
    println!("cargo:rustc-link-lib=static=mlxc");

    // Link the C++ Engine (mlx)
    println!("cargo:rustc-link-search=native={}/build/_deps/mlx-build", manifest_dir.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // SYSTEM FRAMEWORKS 
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // BINDGEN (Generates fresh bindings for the tester's machine)
    let header_path = manifest_dir.join("mlx/c/mlx.h");
    println!("cargo:rerun-if-changed={}", header_path.display());

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .clang_arg(format!("-I{}", manifest_dir.display())) 
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}