// fn main() {
//     // Link the `torch_cpu` library
//     println!("cargo:rustc-link-lib=dylib=torch_cpu");

//     // Optionally, specify the path to libtorch if it's in a non-standard location
//     // Replace `/path/to/libtorch/lib` with the actual path
//     println!("cargo:rustc-link-search=native=/home/snehal/libtorch/lib");

//     // Optionally specify the include directory for libtorch headers
//     // Replace `/path/to/libtorch/include` with the actual path
//     println!("cargo:include=/home/snehal/libtorch/include");
// }