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
    if !Path::new("clip.cpp/ggml/LICENSE").exists() {
        eprintln!("Setting up submodules");
        run_command_or_fail(
            "../clip_cpp-sys/clip.cpp",
            "git",
            &["submodule", "update", "--init"],
        );
    }
    eprintln!("Building ggml statically");
    build_ggml_library();
    eprintln!("Building clip.cpp statically");
    build_clip_cpp_library();
}

fn build_ggml_library() {
    let mut config = cmake::Config::new("clip.cpp/ggml");

    config.no_build_target(true);

    if env::var("CARGO_FEATURE_GGML_STATIC").is_ok() {
        config.define("GGML_STATIC", "1");
    } else {
        config.define("GGML_STATIC", "0");
        if env::var("CARGO_FEATURE_BUILD_SHARED_LIBS").is_ok() {
            config.define("BUILD_SHARED_LIBS", "1");
        } else {
            config.define("BUILD_SHARED_LIBS", "0");
        }
    }

    if env::var("CARGO_FEATURE_ALL_WARNINGS").is_ok() {
        config.define("GGML_ALL_WARNINGS", "1");
    } else {
        config.define("GGML_ALL_WARNINGS", "0");
    }

    if env::var("CARGO_FEATURE_ALL_WARNINGS_3RD_PARTY").is_ok() {
        config.define("GGML_ALL_WARNINGS_3RD_PARTY", "1");
    } else {
        config.define("GGML_ALL_WARNINGS_3RD_PARTY", "0");
    }

    if env::var("CARGO_FEATURE_SANITIZE_THREAD").is_ok() {
        config.define("GGML_SANITIZE_THREAD", "1");
    } else {
        config.define("GGML_SANITIZE_THREAD", "0");
    }

    if env::var("CARGO_FEATURE_SANITIZE_ADDRESS").is_ok() {
        config.define("GGML_SANITIZE_ADDRESS", "1");
    } else {
        config.define("GGML_SANITIZE_ADDRESS", "0");
    }

    if env::var("CARGO_FEATURE_SANITIZE_UNDEFINED").is_ok() {
        config.define("GGML_SANITIZE_UNDEFINED", "1");
    } else {
        config.define("GGML_SANITIZE_UNDEFINED", "0");
    }

    if env::var("CARGO_FEATURE_BUILD_TESTS").is_ok() {
        config.define("GGML_BUILD_TESTS", "1");
    } else {
        config.define("GGML_BUILD_TESTS", "0");
    }

    if env::var("CARGO_FEATURE_BUILD_EXAMPLES").is_ok() {
        config.define("GGML_BUILD_EXAMPLES", "1");
    } else {
        config.define("GGML_BUILD_EXAMPLES", "0");
    }

    if env::var("CARGO_FEATURE_TEST_COVERAGE").is_ok() {
        config.define("GGML_TEST_COVERAGE", "1");
    } else {
        config.define("GGML_TEST_COVERAGE", "0");
    }

    if env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
        config.define("GGML_NO_ACCELERATE", "0");
    } else {
        config.define("GGML_NO_ACCELERATE", "1");
    }

    if env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
        config.define("CLIP_OPENBLAS", "1");
    } else {
        config.define("CLIP_OPENBLAS", "0");
    }

    if env::var("CARGO_FEATURE_GGML_CUBLAS").is_ok() {
        config.define("GGML_CUBLAS", "1");
    } else {
        config.define("GGML_CUBLAS", "0");
    }

    if env::var("CARGO_FEATURE_GGML_OPENBLAS").is_ok() {
        config.define("GGML_OPENBLAS", "1");
    } else {
        config.define("GGML_OPENBLAS", "0");
    }

    if env::var("CARGO_FEATURE_GGML_CUDAF16").is_ok() {
        config.define("GGML_CUDA_F16", "1");
    } else {
        config.define("GGML_CUDA_F16", "0");
    }

    if env::var("CARGO_FEATURE_AVX").is_ok() {
        config.define("GGML_AVX", "1");
    } else {
        config.define("GGML_AVX", "0");
    }

    if env::var("CARGO_FEATURE_AVX2").is_ok() {
        config.define("GGML_AVX2", "1");
    } else {
        config.define("GGML_AVX2", "0");
    }

    if env::var("CARGO_FEATURE_FMA").is_ok() {
        config.define("GGML_FMA", "1");
    } else {
        config.define("GGML_FMA", "0");
    }

    if env::var("CARGO_FEATURE_AVX512").is_ok() {
        config.define("GGML_AVX512", "1");
    } else {
        config.define("GGML_AVX512", "0");
    }

    if env::var("CARGO_FEATURE_AVX512_VBMI").is_ok() {
        config.define("GGML_AVX512_VBMI", "1");
    } else {
        config.define("GGML_AVX512_VBMI", "0");
    }

    if env::var("CARGO_FEATURE_AVX512_VNNI").is_ok() {
        config.define("GGML_AVX512_VNNI", "1");
    } else {
        config.define("GGML_AVX512_VNNI", "0");
    }

    if let Ok(system_name) = env::var("CMAKE_SYSTEM_NAME") {
        config.define("CMAKE_SYSTEM_NAME", system_name);
    }

    println!("Configuring and compiling ggml library");
    let dst = config.build();

    println!(
        "cargo:rustc-link-search=native={}/build/ggml/src",
        dst.display()
    );
    if env::var("CARGO_FEATURE_GGML_CUBLAS").is_ok() {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");

    if env::var("CARGO_FEATURE_GGML_CUBLAS").is_ok()
        && env::var("CARGO_FEATURE_GGML_STATIC").is_ok()
    {
        println!("cargo:rustc-link-lib=static=cudart_static");
        println!("cargo:rustc-link-lib=static=cublas_static");
        println!("cargo:rustc-link-lib=static=cublasLt_static");
    }

    if env::var("CARGO_FEATURE_GGML_STATIC").is_ok() {
        println!("cargo:rustc-link-lib=static=ggml");
    } else {
        println!("cargo:rustc-link-lib=ggml");
    }
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

fn build_clip_cpp_library() {
    let mut config = cmake::Config::new("clip.cpp");

    config.no_build_target(true);

    if env::var("CARGO_FEATURE_STATIC").is_ok() {
        config.define("CLIP_STATIC", "1");
        config.define("STB_IMAGE_STATIC", "1");
    } else {
        config.define("CLIP_STATIC", "0");
        if env::var("CARGO_FEATURE_BUILD_SHARED_LIBS").is_ok() {
            config.define("BUILD_SHARED_LIBS", "1");
        } else {
            config.define("BUILD_SHARED_LIBS", "0");
        }
    }

    if env::var("CARGO_FEATURE_SHARED").is_ok() {
        config.define("CLIP_SHARED", "1");
    } else {
        config.define("CLIP_SHARED", "0");
    }

    if env::var("CARGO_FEATURE_BUILD_TESTS").is_ok() {
        config.define("CLIP_BUILD_TESTS", "1");
    } else {
        config.define("CLIP_BUILD_TESTS", "0");
    }

    if env::var("CARGO_FEATURE_BUILD_EXAMPLES").is_ok() {
        config.define("CLIP_BUILD_EXAMPLES", "1");
    } else {
        config.define("CLIP_BUILD_EXAMPLES", "0");
    }

    if env::var("CARGO_FEATURE_BUILD_IMAGE_SEARCH").is_ok() {
        config.define("CLIP_BUILD_IMAGE_SEARCH", "1");
    } else {
        config.define("CLIP_BUILD_IMAGE_SEARCH", "0");
    }

    if env::var("CARGO_FEATURE_NATIVE").is_ok() {
        config.define("CLIP_NATIVE", "1");
    } else {
        config.define("CLIP_NATIVE", "0");
    }

    if env::var("CARGO_FEATURE_LTO").is_ok() {
        config.define("CLIP_LTO", "1");
    } else {
        config.define("CLIP_LTO", "0");
    }

    if env::var("CARGO_FEATURE_ALL_WARNINGS").is_ok() {
        config.define("CLIP_ALL_WARNINGS", "1");
    } else {
        config.define("CLIP_ALL_WARNINGS", "0");
    }

    if env::var("CARGO_FEATURE_ALL_WARNINGS_3RD_PARTY").is_ok() {
        config.define("CLIP_ALL_WARNINGS_3RD_PARTY", "1");
    } else {
        config.define("CLIP_ALL_WARNINGS_3RD_PARTY", "0");
    }

    if env::var("CARGO_FEATURE_GPROF").is_ok() {
        config.define("CLIP_GPROF", "1");
    } else {
        config.define("CLIP_GPROF", "0");
    }

    if env::var("CARGO_FEATURE_SANITIZE_THREAD").is_ok() {
        config.define("CLIP_SANITIZE_THREAD", "1");
    } else {
        config.define("CLIP_SANITIZE_THREAD", "0");
    }

    if env::var("CARGO_FEATURE_SANITIZE_ADDRESS").is_ok() {
        config.define("CLIP_SANITIZE_ADDRESS", "1");
    } else {
        config.define("CLIP_SANITIZE_ADDRESS", "0");
    }

    if env::var("CARGO_FEATURE_SANITIZE_UNDEFINED").is_ok() {
        config.define("CLIP_SANITIZE_UNDEFINED", "1");
    } else {
        config.define("CLIP_SANITIZE_UNDEFINED", "0");
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

    if env::var("CARGO_FEATURE_FMA").is_ok() {
        config.define("CLIP_FMA", "1");
    } else {
        config.define("CLIP_FMA", "0");
    }

    if env::var("CARGO_FEATURE_AVX512").is_ok() {
        config.define("CLIP_AVX512", "1");
    } else {
        config.define("CLIP_AVX512", "0");
    }

    if env::var("CARGO_FEATURE_AVX512_VBMI").is_ok() {
        config.define("CLIP_AVX512_VBMI", "1");
    } else {
        config.define("CLIP_AVX512_VBMI", "0");
    }

    if env::var("CARGO_FEATURE_AVX512_VNNI").is_ok() {
        config.define("CLIP_AVX512_VNNI", "1");
    } else {
        config.define("CLIP_AVX512_VNNI", "0");
    }

    if env::var("CARGO_FEATURE_F16C").is_ok() {
        config.define("CLIP_F16C", "1");
    } else {
        config.define("CLIP_F16C", "0");
    }

    if env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
        config.define("CLIP_ACCELERATE", "1");
    } else {
        config.define("CLIP_ACCELERATE", "0");
    }

    if env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
        config.define("CLIP_OPENBLAS", "1");
    } else {
        config.define("CLIP_OPENBLAS", "0");
    }

    if env::var("CARGO_FEATURE_GGML_CUBLAS").is_ok() {
        config.define("GGML_CUBLAS", "1");
    } else {
        config.define("GGML_CUBLAS", "0");
    }

    if let Ok(system_name) = env::var("CMAKE_SYSTEM_NAME") {
        config.define("CMAKE_SYSTEM_NAME", system_name);
    }

    println!("Configuring and compiling clip.cpp library");
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dylib=stdc++");
    if env::var("CARGO_FEATURE_STATIC").is_ok() {
        println!("cargo:rustc-link-lib=static=clip");
    } else {
        println!("cargo:rustc-link-lib=clip");
    }
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
