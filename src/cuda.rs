//! Backend CUDA opcional (feature `cuda`).
//!
//! Acelera las matmuls grandes con cuBLAS. Las matrices viven en el host y se
//! copian por PCIe en cada llamada; por eso solo se usa a partir de un umbral
//! de trabajo y el resto de operaciones (atención, capas normales, AdamW)
//! siguen en CPU.

use crate::error::{Error, Result};

/// Trabajo mínimo (m*k*n) para enviar una matmul a la GPU.
pub const GPU_WORK_THRESHOLD: usize = 1 << 20;

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::{c_char, c_float, c_int};

    extern "C" {
        pub fn mini_gpt_cuda_matmul(
            a: *const c_float,
            b: *const c_float,
            c: *mut c_float,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        pub fn mini_gpt_cuda_device_name() -> *const c_char;
    }
}

/// Indica si el binario se compiló con `--features cuda`.
pub fn compiled() -> bool {
    cfg!(feature = "cuda")
}

/// Indica si las operaciones deben ir a la GPU en tiempo de ejecución.
///
/// `MINI_GPT_DEVICE=cpu` permite comparar contra la ruta CPU con el mismo
/// binario; `MINI_GPT_DEVICE=cuda` fuerza GPU cuando está compilada.
pub fn enabled() -> bool {
    #[cfg(feature = "cuda")]
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| match std::env::var("MINI_GPT_DEVICE") {
            Ok(value) => !value.eq_ignore_ascii_case("cpu"),
            Err(_) => true,
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Nombre del dispositivo CUDA, o "CPU" si no hay backend.
pub fn device_name() -> String {
    if !enabled() {
        return "CPU".to_string();
    }
    #[cfg(feature = "cuda")]
    {
        let pointer = unsafe { ffi::mini_gpt_cuda_device_name() };
        if pointer.is_null() {
            return "GPU".to_string();
        }
        unsafe { std::ffi::CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
    #[cfg(not(feature = "cuda"))]
    {
        "CPU".to_string()
    }
}

/// Descripción corta del backend para los logs de la CLI.
pub fn backend_description() -> String {
    if enabled() {
        format!("CUDA ({})", device_name())
    } else if compiled() {
        "CPU (CUDA compilada pero desactivada con MINI_GPT_DEVICE=cpu)".to_string()
    } else {
        "CPU".to_string()
    }
}

/// C = A (m x k) * B (k x n), en fila mayor, ejecutado en la GPU.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    if a.len() != m * k || b.len() != k * n {
        return Err(Error::new("Dimensiones incompatibles para matmul CUDA"));
    }
    #[cfg(feature = "cuda")]
    {
        let mut output = vec![0.0f32; m * n];
        let status = unsafe {
            ffi::mini_gpt_cuda_matmul(
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
                m as i32,
                k as i32,
                n as i32,
            )
        };
        if status != 0 {
            return Err(Error::new(format!("matmul CUDA falló con código {status}")));
        }
        Ok(output)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, m, k, n);
        Err(Error::new(
            "El backend CUDA no está compilado (usa --features cuda)",
        ))
    }
}

/// Intenta la matmul en GPU y devuelve `None` si no aplica o si falla.
pub fn try_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    if !enabled() || m * k * n < GPU_WORK_THRESHOLD {
        return None;
    }
    match matmul(a, b, m, k, n) {
        Ok(values) => Some(values),
        Err(error) => {
            warn_once(&error.to_string());
            None
        }
    }
}

fn warn_once(message: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("aviso: fallo en CUDA, se continúa en CPU: {message}");
    }
}
