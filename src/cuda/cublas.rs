//! Multiplicación FP32 con cuBLAS.
//!
//! La API acepta buffers row-major del motor CPU. Se invierten los operandos
//! y las dimensiones al llamar a SGEMM porque cuBLAS interpreta matrices
//! column-major. Puede usar el stream predeterminado o uno explícito asociado
//! mediante `CudaContext::create_blas_on_stream`.

use super::{dynamic, CudaBuffer, CudaContext, ContextInner, Error, Result};
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;
use std::sync::Arc;

const CUBLAS_SUCCESS: c_int = 0;
const CUBLAS_OP_N: c_int = 0;

type CublasCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type CublasDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type CublasSetStream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type CublasSgemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> c_int;

/// Handle cuBLAS ligado al contexto CUDA que lo creó.
pub struct CudaBlas<'stream> {
    context: Arc<ContextInner>,
    handle: NonNull<c_void>,
    _library: dynamic::Library,
    destroy: CublasDestroy,
    sgemm: CublasSgemm,
    // Keep the stream alive for at least as long as cuBLAS can enqueue work on it.
    _stream: Option<&'stream super::CudaStream>,
}

impl CudaContext {
    /// Carga cuBLAS dinámicamente y crea un handle para el stream predeterminado.
    pub fn create_blas(&self) -> Result<CudaBlas<'static>> {
        self.create_blas_internal(None)
    }

    /// Crea un handle cuBLAS asociado a un stream explícito.
    ///
    /// El handle conserva un préstamo del stream, impidiendo destruirlo antes
    /// de que deje de usarse cuBLAS. Los buffers deben seguir perteneciendo al
    /// mismo contexto.
    pub fn create_blas_on_stream<'stream>(
        &self,
        stream: &'stream super::CudaStream,
    ) -> Result<CudaBlas<'stream>> {
        if !Arc::ptr_eq(&self.inner, &stream.context) {
            return Err(Error::new("El stream cuBLAS debe pertenecer al mismo contexto CUDA"));
        }
        self.create_blas_internal(Some(stream))
    }

    fn create_blas_internal<'stream>(
        &self,
        stream: Option<&'stream super::CudaStream>,
    ) -> Result<CudaBlas<'stream>> {
        self.inner.make_current()?;
        let library = dynamic::Library::open_candidates(
            &["libcublas.so.12", "libcublas.so.13", "libcublas.so"],
            "No se pudo cargar libcublas. Instala el runtime cuBLAS compatible con el driver NVIDIA",
        )?;
        let create: CublasCreate = unsafe { library.symbol("cublasCreate_v2")? };
        let destroy: CublasDestroy = unsafe { library.symbol("cublasDestroy_v2")? };
        let set_stream: CublasSetStream = unsafe { library.symbol("cublasSetStream_v2")? };
        let sgemm: CublasSgemm = unsafe { library.symbol("cublasSgemm_v2")? };

        let mut raw_handle = std::ptr::null_mut();
        check_cublas(unsafe { create(&mut raw_handle) }, "cublasCreate_v2")?;
        let handle = NonNull::new(raw_handle)
            .ok_or_else(|| Error::new("cuBLAS devolvió un handle nulo"))?;

        if let Some(stream) = stream {
            let status = unsafe { set_stream(handle.as_ptr(), stream.raw.as_ptr()) };
            if let Err(error) = check_cublas(status, "cublasSetStream_v2") {
                unsafe { destroy(handle.as_ptr()); }
                return Err(error);
            }
        }

        Ok(CudaBlas {
            context: Arc::clone(&self.inner),
            handle,
            _library: library,
            destroy,
            sgemm,
            _stream: stream,
        })
    }
}

impl CudaBlas<'_> {
    /// Calcula C = A × B para matrices FP32 row-major.
    ///
    /// Dimensiones: A=[rows, inner], B=[inner, columns], C=[rows, columns].
    /// No copia buffers a RAM ni sincroniza automáticamente.
    pub fn matmul_row_major(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        c: &mut CudaBuffer,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<()> {
        if rows == 0 || inner == 0 || columns == 0 {
            return Err(Error::new("Las dimensiones de matmul CUDA deben ser mayores que cero"));
        }
        let a_len = rows.checked_mul(inner)
            .ok_or_else(|| Error::new("Dimensiones de A fuera de rango"))?;
        let b_len = inner.checked_mul(columns)
            .ok_or_else(|| Error::new("Dimensiones de B fuera de rango"))?;
        let c_len = rows.checked_mul(columns)
            .ok_or_else(|| Error::new("Dimensiones de C fuera de rango"))?;
        if a.len() != a_len || b.len() != b_len || c.len() != c_len {
            return Err(Error::new(format!(
                "Tamaños incompatibles en matmul CUDA: A={} (esperado {}), B={} (esperado {}), C={} (esperado {})",
                a.len(), a_len, b.len(), b_len, c.len(), c_len
            )));
        }
        if !Arc::ptr_eq(&self.context, &a.context)
            || !Arc::ptr_eq(&self.context, &b.context)
            || !Arc::ptr_eq(&self.context, &c.context)
        {
            return Err(Error::new("Todos los buffers de matmul deben pertenecer al mismo contexto CUDA"));
        }
        let (m, n, k, lda, ldb, ldc) = row_major_sgemm_dimensions(rows, inner, columns);
        if [m, n, k].iter().any(|&value| value > c_int::MAX as usize) {
            return Err(Error::new("Dimensiones de matmul superan el límite de cuBLAS"));
        }

        self.context.make_current()?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // A_row[m,k] y B_row[k,n] se ven como A_col[k,m] y B_col[n,k].
        // C_row[m,n] se ve como C_col[n,m], por eso SGEMM calcula B_col × A_col.
        let status = unsafe {
            (self.sgemm)(
                self.handle.as_ptr(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m as c_int,
                n as c_int,
                k as c_int,
                &alpha,
                b.pointer as *const f32,
                m as c_int,
                a.pointer as *const f32,
                k as c_int,
                &beta,
                c.pointer as *mut f32,
                m as c_int,
            )
        };
        check_cublas(status, "cublasSgemm_v2")
    }
}

impl Drop for CudaBlas<'_> {
    fn drop(&mut self) {
        if self.context.make_current().is_ok() {
            unsafe {
                (self.destroy)(self.handle.as_ptr());
            }
        }
    }
}

fn check_cublas(status: c_int, operation: &str) -> Result<()> {
    if status == CUBLAS_SUCCESS {
        Ok(())
    } else {
        Err(Error::new(format!(
            "cuBLAS: {operation} falló con código {status}"
        )))
    }
}

/// Convierte dimensiones row-major a la llamada column-major de cuBLAS.
fn row_major_sgemm_dimensions(
    rows: usize,
    inner: usize,
    columns: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    // SGEMM: B_col[columns, inner] × A_col[inner, rows] = C_col[columns, rows].
    (columns, rows, inner, columns, inner, columns)
}

#[cfg(test)]
mod tests {
    use super::row_major_sgemm_dimensions;

    #[test]
    fn row_major_dimensions_are_transposed_for_cublas() {
        assert_eq!(
            row_major_sgemm_dimensions(2, 3, 4),
            (4, 2, 3, 4, 3, 4)
        );
    }
}
