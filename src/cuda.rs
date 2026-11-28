//! Descubrimiento del dispositivo y gestión inicial del contexto CUDA.
//!
//! Esta primera capa usa la CUDA Driver API mediante carga dinámica: compilar
//! con `--features cuda` no requiere instalar el CUDA Toolkit ni enlazar
//! bibliotecas NVIDIA. El driver NVIDIA sí debe estar instalado al ejecutar.
//!
//! Incluye operaciones tensoriales FP32, cuBLAS y kernels NVRTC; el GPT completo
//! todavía no está conectado a este backend.

use crate::error::{Error, Result};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr::NonNull;
use std::sync::Arc;

const CUDA_SUCCESS: c_int = 0;

pub mod cublas;
pub mod kernels;
pub mod tensor;

/// Información consultada directamente al CUDA Driver API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub total_memory_bytes: u64,
    pub driver_version: i32,
}

impl CudaDeviceInfo {
    /// Etiqueta de arquitectura de NVIDIA, por ejemplo `sm_89` para Ada.
    pub fn architecture(&self) -> String {
        architecture_tag(self.compute_major, self.compute_minor)
    }

    /// Versión legible del driver, por ejemplo `12.8`.
    pub fn driver_version_string(&self) -> String {
        driver_version_string(self.driver_version)
    }
}

/// Runtime CUDA cargado dinámicamente y lista de dispositivos disponibles.
pub struct CudaRuntime {
    api: Arc<DriverApi>,
    devices: Vec<CudaDeviceInfo>,
}

impl CudaRuntime {
    /// Carga el driver NVIDIA, inicializa CUDA y consulta todos los dispositivos.
    pub fn discover() -> Result<Self> {
        let api = Arc::new(unsafe { DriverApi::load()? });
        check_cuda(unsafe { (api.init)(0) }, "cuInit")?;

        let mut count = 0;
        check_cuda(unsafe { (api.device_get_count)(&mut count) }, "cuDeviceGetCount")?;
        if count <= 0 {
            return Err(Error::new("CUDA está instalado, pero no hay dispositivos NVIDIA disponibles"));
        }

        let mut devices = Vec::with_capacity(count as usize);
        let mut driver_version = 0;
        check_cuda(
            unsafe { (api.driver_get_version)(&mut driver_version) },
            "cuDriverGetVersion",
        )?;

        for ordinal in 0..count {
            let mut device = 0;
            check_cuda(
                unsafe { (api.device_get)(&mut device, ordinal) },
                "cuDeviceGet",
            )?;

            let mut name = [0 as c_char; 256];
            check_cuda(
                unsafe { (api.device_get_name)(name.as_mut_ptr(), name.len() as c_int, device) },
                "cuDeviceGetName",
            )?;
            let name = unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            let (mut major, mut minor) = (0, 0);
            check_cuda(
                unsafe { (api.compute_capability)(&mut major, &mut minor, device) },
                "cuDeviceComputeCapability",
            )?;

            let mut total_memory = 0usize;
            check_cuda(
                unsafe { (api.device_total_memory)(&mut total_memory, device) },
                "cuDeviceTotalMem_v2",
            )?;

            devices.push(CudaDeviceInfo {
                ordinal: ordinal as usize,
                name,
                compute_major: major as u32,
                compute_minor: minor as u32,
                total_memory_bytes: total_memory as u64,
                driver_version,
            });
        }

        Ok(Self { api, devices })
    }

    pub fn devices(&self) -> &[CudaDeviceInfo] {
        &self.devices
    }

    pub fn device(&self, ordinal: usize) -> Result<&CudaDeviceInfo> {
        self.devices.get(ordinal).ok_or_else(|| {
            Error::new(format!(
                "No existe el dispositivo CUDA {ordinal}; dispositivos detectados: {}",
                self.devices.len()
            ))
        })
    }

    /// Crea un contexto CUDA asociado al ordinal solicitado.
    ///
    /// El contexto se destruye automáticamente cuando sale de ámbito.
    pub fn create_context(&self, ordinal: usize) -> Result<CudaContext> {
        self.device(ordinal)?;
        let mut raw_context = std::ptr::null_mut();
        let mut device = 0;
        check_cuda(
            unsafe { (self.api.device_get)(&mut device, ordinal as c_int) },
            "cuDeviceGet",
        )?;
        check_cuda(
            unsafe { (self.api.context_create)(&mut raw_context, 0, device) },
            "cuCtxCreate_v2",
        )?;
        let context = NonNull::new(raw_context)
            .ok_or_else(|| Error::new("CUDA devolvió un contexto nulo"))?;

        Ok(CudaContext {
            inner: Arc::new(ContextInner {
                api: Arc::clone(&self.api),
                raw: context,
            }),
            ordinal,
        })
    }
}

/// Contexto CUDA RAII. El contexto permanece vivo mientras existan buffers.
pub struct CudaContext {
    inner: Arc<ContextInner>,
    ordinal: usize,
}

struct ContextInner {
    api: Arc<DriverApi>,
    raw: NonNull<c_void>,
}

impl ContextInner {
    fn make_current(&self) -> Result<()> {
        check_cuda(
            unsafe { (self.api.context_set_current)(self.raw.as_ptr()) },
            "cuCtxSetCurrent",
        )
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // La destrucción es best-effort: Drop no puede propagar errores.
        unsafe {
            (self.api.context_destroy)(self.raw.as_ptr());
        }
    }
}

impl CudaContext {
    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    /// Reserva memoria de dispositivo para un vector FP32 residente.
    pub fn allocate_f32(&self, len: usize) -> Result<CudaBuffer> {
        if len == 0 {
            return Err(Error::new("No se pueden reservar buffers CUDA vacíos"));
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::new("Tamaño de buffer CUDA fuera de rango"))?;
        self.inner.make_current()?;
        let mut pointer = 0u64;
        check_cuda(
            unsafe { (self.inner.api.mem_alloc)(&mut pointer, bytes) },
            "cuMemAlloc_v2",
        )?;
        if pointer == 0 {
            return Err(Error::new("CUDA devolvió un puntero de dispositivo nulo"));
        }
        Ok(CudaBuffer {
            pointer,
            len,
            context: Arc::clone(&self.inner),
        })
    }

    /// Crea un stream no predeterminado para ordenar trabajo en GPU.
    pub fn create_stream(&self) -> Result<CudaStream> {
        self.inner.make_current()?;
        let mut raw_stream = std::ptr::null_mut();
        check_cuda(
            unsafe { (self.inner.api.stream_create)(&mut raw_stream, 0) },
            "cuStreamCreate",
        )?;
        let stream = NonNull::new(raw_stream)
            .ok_or_else(|| Error::new("CUDA devolvió un stream nulo"))?;
        Ok(CudaStream {
            context: Arc::clone(&self.inner),
            raw: stream,
        })
    }

    /// Espera a que termine todo el trabajo del contexto actual.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.make_current()?;
        check_cuda(
            unsafe { (self.inner.api.context_synchronize)() },
            "cuCtxSynchronize",
        )
    }
}

/// Stream CUDA propietario. Su ciclo de vida está ligado al contexto.
pub struct CudaStream {
    context: Arc<ContextInner>,
    raw: NonNull<c_void>,
}

impl CudaStream {
    /// Espera únicamente al trabajo encolado en este stream.
    pub fn synchronize(&self) -> Result<()> {
        self.context.make_current()?;
        check_cuda(
            unsafe { (self.context.api.stream_synchronize)(self.raw.as_ptr()) },
            "cuStreamSynchronize",
        )
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.context.make_current().is_ok() {
            unsafe {
                (self.context.api.stream_destroy)(self.raw.as_ptr());
            }
        }
    }
}

/// Buffer FP32 propietario en VRAM. No descarga datos implícitamente.
pub struct CudaBuffer {
    pointer: u64,
    len: usize,
    context: Arc<ContextInner>,
}

impl CudaBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copia un vector completo desde RAM a VRAM.
    pub fn copy_from_host(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.len {
            return Err(Error::new(format!(
                "Tamaño H2D incorrecto: buffer {}, origen {}",
                self.len,
                values.len()
            )));
        }
        let bytes = self.len * std::mem::size_of::<f32>();
        self.context.make_current()?;
        check_cuda(
            unsafe {
                (self.context.api.memcpy_htod)(
                    self.pointer,
                    values.as_ptr().cast::<c_void>(),
                    bytes,
                )
            },
            "cuMemcpyHtoD_v2",
        )
    }

    /// Copia el contenido completo de VRAM a RAM de forma explícita.
    pub fn copy_to_host(&self) -> Result<Vec<f32>> {
        let mut values = vec![0.0f32; self.len];
        let bytes = self.len * std::mem::size_of::<f32>();
        self.context.make_current()?;
        check_cuda(
            unsafe {
                (self.context.api.memcpy_dtoh)(
                    values.as_mut_ptr().cast::<c_void>(),
                    self.pointer,
                    bytes,
                )
            },
            "cuMemcpyDtoH_v2",
        )?;
        Ok(values)
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // El buffer conserva el contexto mediante Arc; liberar es best-effort.
        if self.context.make_current().is_ok() {
            unsafe {
                (self.context.api.mem_free)(self.pointer);
            }
        }
    }
}

fn architecture_tag(major: u32, minor: u32) -> String {
    format!("sm_{major}{minor}")
}

fn driver_version_string(version: i32) -> String {
    if version <= 0 {
        return "desconocida".to_string();
    }
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn check_cuda(status: c_int, operation: &str) -> Result<()> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::new(format!(
            "CUDA Driver API: {operation} falló con código {status}"
        )))
    }
}

type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type CuDeviceComputeCapability = unsafe extern "C" fn(*mut c_int, *mut c_int, c_int) -> c_int;
type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, c_int) -> c_int;
type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> c_int;
type CuContextCreate = unsafe extern "C" fn(*mut *mut c_void, c_uint, c_int) -> c_int;
type CuContextDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type CuContextSetCurrent = unsafe extern "C" fn(*mut c_void) -> c_int;
type CuContextSynchronize = unsafe extern "C" fn() -> c_int;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> c_int;
type CuMemFree = unsafe extern "C" fn(u64) -> c_int;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> c_int;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int;
type CuStreamCreate = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> c_int;
type CuStreamDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type CuStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> c_int;
type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> c_int;
type CuModuleGetFunction = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int;
type CuLaunchKernel = unsafe extern "C" fn(
    *mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
    *mut c_void, *mut *mut c_void, *mut *mut c_void
) -> c_int;

struct DriverApi {
    _library: dynamic::Library,
    init: CuInit,
    device_get_count: CuDeviceGetCount,
    device_get: CuDeviceGet,
    device_get_name: CuDeviceGetName,
    compute_capability: CuDeviceComputeCapability,
    device_total_memory: CuDeviceTotalMem,
    driver_get_version: CuDriverGetVersion,
    context_create: CuContextCreate,
    context_destroy: CuContextDestroy,
    context_set_current: CuContextSetCurrent,
    context_synchronize: CuContextSynchronize,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    stream_create: CuStreamCreate,
    stream_destroy: CuStreamDestroy,
    stream_synchronize: CuStreamSynchronize,
    module_load_data: CuModuleLoadData,
    module_unload: CuModuleUnload,
    module_get_function: CuModuleGetFunction,
    launch_kernel: CuLaunchKernel,
}

impl DriverApi {
    unsafe fn load() -> Result<Self> {
        let library = dynamic::Library::open()?;
        Ok(Self {
            init: library.symbol("cuInit")?,
            device_get_count: library.symbol("cuDeviceGetCount")?,
            device_get: library.symbol("cuDeviceGet")?,
            device_get_name: library.symbol("cuDeviceGetName")?,
            compute_capability: library.symbol("cuDeviceComputeCapability")?,
            device_total_memory: library.symbol("cuDeviceTotalMem_v2")?,
            driver_get_version: library.symbol("cuDriverGetVersion")?,
            context_create: library.symbol("cuCtxCreate_v2")?,
            context_destroy: library.symbol("cuCtxDestroy_v2")?,
            context_set_current: library.symbol("cuCtxSetCurrent")?,
            context_synchronize: library.symbol("cuCtxSynchronize")?,
            mem_alloc: library.symbol("cuMemAlloc_v2")?,
            mem_free: library.symbol("cuMemFree_v2")?,
            memcpy_htod: library.symbol("cuMemcpyHtoD_v2")?,
            memcpy_dtoh: library.symbol("cuMemcpyDtoH_v2")?,
            stream_create: library.symbol("cuStreamCreate")?,
            stream_destroy: library.symbol("cuStreamDestroy_v2")?,
            stream_synchronize: library.symbol("cuStreamSynchronize")?,
            module_load_data: library.symbol("cuModuleLoadData")?,
            module_unload: library.symbol("cuModuleUnload")?,
            module_get_function: library.symbol("cuModuleGetFunction")?,
            launch_kernel: library.symbol("cuLaunchKernel")?,
            _library: library,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{architecture_tag, driver_version_string};

    #[test]
    fn formats_ada_compute_capability() {
        assert_eq!(architecture_tag(8, 9), "sm_89");
    }

    #[test]
    fn formats_cuda_driver_version() {
        assert_eq!(driver_version_string(12080), "12.8");
        assert_eq!(driver_version_string(0), "desconocida");
    }
}

#[cfg(target_os = "linux")]
mod dynamic {
    use super::{c_char, c_int, c_void, CString, Error, NonNull, Result};
    use std::ffi::CStr;

    const RTLD_NOW: c_int = 2;

    #[link(name = "dl")]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    pub struct Library {
        handle: NonNull<c_void>,
    }

    impl Library {
        pub fn open() -> Result<Self> {
            Self::open_candidates(
                &["libcuda.so.1", "libcuda.so"],
                "No se pudo cargar libcuda.so.1. Instala un driver NVIDIA compatible y comprueba LD_LIBRARY_PATH",
            )
        }

        pub fn open_candidates(candidates: &[&str], error_message: &str) -> Result<Self> {
            for candidate in candidates {
                let name = CString::new(*candidate).expect("static library name");
                let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
                if let Some(handle) = NonNull::new(handle) {
                    return Ok(Self { handle });
                }
            }
            Err(Error::new(error_message))
        }

        pub unsafe fn symbol<T: Copy>(&self, name: &str) -> Result<T> {
            let name = CString::new(name).map_err(|_| Error::new("Nombre de símbolo CUDA inválido"))?;
            let pointer = dlsym(self.handle.as_ptr(), name.as_ptr());
            if pointer.is_null() {
                return Err(Error::new(format!(
                    "El driver NVIDIA no exporta el símbolo {}",
                    CStr::from_ptr(name.as_ptr()).to_string_lossy()
                )));
            }
            Ok(std::mem::transmute_copy(&pointer))
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            unsafe {
                dlclose(self.handle.as_ptr());
            }
        }
    }
}
