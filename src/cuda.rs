//! Descubrimiento del dispositivo y gestión inicial del contexto CUDA.
//!
//! Esta primera capa usa la CUDA Driver API mediante carga dinámica: compilar
//! con `--features cuda` no requiere instalar el CUDA Toolkit ni enlazar
//! bibliotecas NVIDIA. El driver NVIDIA sí debe estar instalado al ejecutar.
//!
//! Todavía no implementa tensores, kernels ni operaciones de entrenamiento.

use crate::error::{Error, Result};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr::NonNull;
use std::sync::Arc;

const CUDA_SUCCESS: c_int = 0;

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
            api: Arc::clone(&self.api),
            raw: context,
            ordinal,
        })
    }
}

/// Contexto CUDA RAII. Mantiene cargado el driver hasta destruir el contexto.
pub struct CudaContext {
    api: Arc<DriverApi>,
    raw: NonNull<c_void>,
    ordinal: usize,
}

impl CudaContext {
    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // La destrucción es best-effort: Drop no puede propagar errores.
        unsafe {
            (self.api.context_destroy)(self.raw.as_ptr());
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
            for candidate in ["libcuda.so.1", "libcuda.so"] {
                let name = CString::new(candidate).expect("static library name");
                let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
                if let Some(handle) = NonNull::new(handle) {
                    return Ok(Self { handle });
                }
            }
            Err(Error::new(
                "No se pudo cargar libcuda.so.1. Instala un driver NVIDIA compatible y comprueba LD_LIBRARY_PATH",
            ))
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
