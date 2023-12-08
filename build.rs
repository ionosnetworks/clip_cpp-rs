fn main() {
    println!("cargo:rustc-link-search=/opt/faiss/lib");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-search=/opt/homebrew/opt/openblas/lib");
}
