use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = "/Users/ekole/Dev/mlx-c"; // Hardcoded path
    
    // 1. Link the C-Wrapper (The interface)
  
    println!("cargo:rustc-link-search=native={}/build", manifest_dir);
    println!("cargo:rustc-link-lib=static=mlxc");

    // 2. Link the C++ Engine (The backend)
    println!("cargo:rustc-link-search=native={}/build/_deps/mlx-build", manifest_dir);
    println!("cargo:rustc-link-lib=static=mlx");

    // 3. Link necessary system libraries (Metal, Foundation, Accelerate)
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
// 4. Tell Cargo to rerun if the header changes
    let header_path = format!("{}/mlx/c/mlx.h", manifest_dir);
    println!("cargo:rerun-if-changed={}", header_path);

    // 5. Configure Bindgen
    let bindings = bindgen::Builder::default()
        .header(header_path)
        .clang_arg(format!("-I{}", manifest_dir)) 
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // 6. Write the bindings
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
}