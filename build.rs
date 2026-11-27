use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MINI_GPT_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc = find_nvcc().expect(
        "No se encontró nvcc. Instala el CUDA Toolkit o define NVCC=/ruta/a/nvcc \
         (compila sin --features cuda para usar solo CPU).",
    );
    let architecture = cuda_architecture(&nvcc);
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let object = out_dir.join("kernels.o");
    let archive = out_dir.join("libmini_gpt_cuda.a");

    let status = Command::new(&nvcc)
        .args([
            "-O3",
            "--std=c++17",
            "-Xcompiler",
            "-fPIC",
            "-c",
            "cuda/kernels.cu",
            "-o",
            object.to_str().unwrap(),
            &format!("-arch={architecture}"),
        ])
        .status()
        .expect("No se pudo ejecutar nvcc");
    assert!(status.success(), "nvcc falló al compilar cuda/kernels.cu");

    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = Command::new(&ar)
        .args(["crs", archive.to_str().unwrap(), object.to_str().unwrap()])
        .status()
        .expect("No se pudo ejecutar ar");
    assert!(status.success(), "ar falló al crear la librería CUDA");

    let cuda_lib_dir = find_cuda_lib_dir(&nvcc);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=static=mini_gpt_cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib_dir.display());
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os("NVCC") {
        return Some(PathBuf::from(value));
    }
    for candidate in ["/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    let status = Command::new("nvcc").arg("--version").output().ok()?;
    if status.status.success() {
        Some(PathBuf::from("nvcc"))
    } else {
        None
    }
}

fn cuda_architecture(nvcc: &Path) -> String {
    if let Ok(value) = std::env::var("MINI_GPT_CUDA_ARCH") {
        return value;
    }
    // RTX 4060 (Ada) es compute capability 8.9.
    if let Ok(output) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(first) = text.lines().next() {
                let capability = first.trim().replace('.', "");
                if !capability.is_empty() && capability.chars().all(|value| value.is_ascii_digit())
                {
                    return format!("sm_{capability}");
                }
            }
        }
    }
    let _ = nvcc;
    "sm_89".to_string()
}

fn find_cuda_lib_dir(nvcc: &Path) -> PathBuf {
    if let Some(value) = std::env::var_os("CUDA_LIB_DIR") {
        return PathBuf::from(value);
    }
    if let Some(root) = std::env::var_os("CUDA_HOME").or_else(|| std::env::var_os("CUDA_PATH")) {
        let candidate = PathBuf::from(root).join("lib64");
        if candidate.exists() {
            return candidate;
        }
    }
    if let Some(parent) = nvcc.parent().and_then(Path::parent) {
        let candidate = parent.join("lib64");
        if candidate.exists() {
            return candidate;
        }
    }
    for candidate in [
        "/opt/cuda/lib64",
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("/usr/local/cuda/lib64")
}
