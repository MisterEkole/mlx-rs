use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = "/Users/ekole/Dev/mlx-c"; 
    
  
  
    println!("cargo:rustc-link-search=native={}/build", manifest_dir);
    println!("cargo:rustc-link-lib=static=mlxc");

   
    println!("cargo:rustc-link-search=native={}/build/_deps/mlx-build", manifest_dir);
    println!("cargo:rustc-link-lib=static=mlx");

    
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    let header_path = format!("{}/mlx/c/mlx.h", manifest_dir);
    println!("cargo:rerun-if-changed={}", header_path);

   
    let bindings = bindgen::Builder::default()
        .header(header_path)
        .clang_arg(format!("-I{}", manifest_dir)) 
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
}