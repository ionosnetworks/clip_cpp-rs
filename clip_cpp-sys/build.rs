use std::borrow::Borrow;
use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run_command_or_fail<P, S>(dir: &str, cmd: P, args: &[S])
where
    P: AsRef<Path>,
    S: Borrow<str> + AsRef<OsStr>,
{
    let cmd = cmd.as_ref();
    let cmd = if cmd.components().count() > 1 && cmd.is_relative() {
        // If `cmd` is a relative path (and not a bare command that should be
        // looked up in PATH), absolutize it relative to `dir`, as otherwise the
        // behavior of std::process::Command is undefined.
        // https://github.com/rust-lang/rust/issues/37868
        PathBuf::from(dir)
            .join(cmd)
            .canonicalize()
            .expect("canonicalization failed")
    } else {
        PathBuf::from(cmd)
    };
    eprintln!(
        "Running command: \"{} {}\" in dir: {}",
        cmd.display(),
        args.join(" "),
        dir
    );
    let ret = Command::new(cmd).current_dir(dir).args(args).status();
    match ret.map(|status| (status.success(), status.code())) {
        Ok((true, _)) => (),
        Ok((false, Some(c))) => panic!("Command failed with error code {}", c),
        Ok((false, None)) => panic!("Command got killed"),
        Err(e) => panic!("Command failed with error: {}", e),
    }
}

fn main() {
    if !Path::new("clip.cpp/LICENSE").exists() {
        eprintln!("Setting up submodules");
        run_command_or_fail("../", "git", &["submodule", "update", "--init"]);
    }
    eprintln!("Building and linking clip.cpp statically");
    build_clip_cpp_library();
}

fn build_clip_cpp_library() {
    let mut config = cmake::Config::new("clip.cpp");

    config.no_build_target(true);

    if env::var("CARGO_FEATURE_BUILD_IMAGE_SEARCH").is_ok() {
        config.define("CLIP_BUILD_IMAGE_SEARCH", "1");
    } else {
        config.define("CLIP_BUILD_IMAGE_SEARCH", "0");
    }

    config.define("CLIP_NATIVE", "1");

    if env::var("CARGO_FEATURE_GGML_CUBLAS").is_ok() {
        config.define("GGML_CUBLAS", "1");
    } else {
        config.define("GGML_CUBLAS", "0");
    }

    if env::var("CARGO_FEATURE_AVX").is_ok() {
        config.define("CLIP_AVX", "1");
    } else {
        config.define("CLIP_AVX", "0");
    }

    if env::var("CARGO_FEATURE_AVX2").is_ok() {
        config.define("CLIP_AVX2", "1");
    } else {
        config.define("CLIP_AVX2", "0");
    }

    if env::var("CARGO_FEATURE_BUILD_SHARED_LIBS").is_ok() {
        config.define("BUILD_SHARED_LIBS", "1");
    } else {
        config.define("BUILD_SHARED_LIBS", "0");
    }

    if env::var("CARGO_FEATURE_BUILD_TESTS").is_ok() {
        config.define("CLIP_BUILD_TESTS", "1");
    } else {
        config.define("CLIP_BUILD_TESTS", "0");
    }

    if let Ok(system_name) = env::var("CMAKE_SYSTEM_NAME") {
        config.define("CMAKE_SYSTEM_NAME", system_name);
    }

    println!("Configuring and compiling clip.cpp library");
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=clip");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-Iclip.cpp/ggml/include")
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
