//! Compilación NVRTC y lanzamiento de kernels CUDA en tiempo de ejecución.
//!
//! NVRTC compila CUDA C++ a PTX; el CUDA Driver API carga ese PTX y ejecuta
//! funciones sin requerir nvcc ni build.rs.
//! Los argumentos de kernel se pasan como punteros a sus valores ABI; el
//! llamador debe garantizar su validez y el layout esperado por el kernel.

use super::{dynamic, CudaBuffer, CudaContext, CudaStream, ContextInner, Error, Result};
use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr::NonNull;
use std::sync::Arc;

const NVRTC_SUCCESS: c_int = 0;

type NvrtcProgram = *mut c_void;
type NvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram, *const c_char, *const c_char, c_int,
    *const *const c_char, *const *const c_char,
) -> c_int;
type NvrtcCompileProgram = unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> c_int;
type NvrtcGetSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int;
type NvrtcGetText = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int;
type NvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> c_int;

struct NvrtcApi {
    _library: dynamic::Library,
    create_program: NvrtcCreateProgram,
    compile_program: NvrtcCompileProgram,
    get_ptx_size: NvrtcGetSize,
    get_ptx: NvrtcGetText,
    get_log_size: NvrtcGetSize,
    get_log: NvrtcGetText,
    destroy_program: NvrtcDestroyProgram,
}

impl NvrtcApi {
    unsafe fn load() -> Result<Self> {
        let library = dynamic::Library::open_candidates(
            &["libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"],
            "No se pudo cargar NVRTC. Instala el runtime NVIDIA NVRTC compatible",
        )?;
        Ok(Self {
            create_program: library.symbol("nvrtcCreateProgram")?,
            compile_program: library.symbol("nvrtcCompileProgram")?,
            get_ptx_size: library.symbol("nvrtcGetPTXSize")?,
            get_ptx: library.symbol("nvrtcGetPTX")?,
            get_log_size: library.symbol("nvrtcGetProgramLogSize")?,
            get_log: library.symbol("nvrtcGetProgramLog")?,
            destroy_program: library.symbol("nvrtcDestroyProgram")?,
            _library: library,
        })
    }

    fn compile(&self, source: &str, name: &str, architecture: &str) -> Result<Vec<u8>> {
        let source = CString::new(source).map_err(|_| Error::new("El código CUDA contiene un byte NUL"))?;
        let name = CString::new(name).map_err(|_| Error::new("Nombre de programa CUDA inválido"))?;
        let arch_option = CString::new(format!("--gpu-architecture={architecture}"))
            .map_err(|_| Error::new("Arquitectura CUDA inválida"))?;
        let options = [arch_option.as_ptr()];
        let mut program = std::ptr::null_mut();

        check_nvrtc(unsafe {
            (self.create_program)(
                &mut program, source.as_ptr(), name.as_ptr(), 0,
                std::ptr::null(), std::ptr::null(),
            )
        }, "nvrtcCreateProgram")?;

        let result = unsafe { (self.compile_program)(program, 1, options.as_ptr()) };
        if result != NVRTC_SUCCESS {
            let log = self.program_log(program).unwrap_or_else(|_| format!("NVRTC devolvió código {result} sin log"));
            unsafe { (self.destroy_program)(&mut program); }
            return Err(Error::new(format!("Error compilando kernel CUDA:\n{log}")));
        }

        let mut ptx_size = 0usize;
        let size_status = unsafe { (self.get_ptx_size)(program, &mut ptx_size) };
        if let Err(error) = check_nvrtc(size_status, "nvrtcGetPTXSize") {
            unsafe { (self.destroy_program)(&mut program); }
            return Err(error);
        }
        if ptx_size == 0 {
            unsafe { (self.destroy_program)(&mut program); }
            return Err(Error::new("NVRTC generó un PTX vacío"));
        }
        let mut ptx = vec![0u8; ptx_size];
        let status = unsafe { (self.get_ptx)(program, ptx.as_mut_ptr().cast::<c_char>()) };
        unsafe { (self.destroy_program)(&mut program); }
        check_nvrtc(status, "nvrtcGetPTX")?;
        Ok(ptx)
    }

    fn program_log(&self, program: NvrtcProgram) -> Result<String> {
        let mut size = 0usize;
        check_nvrtc(unsafe { (self.get_log_size)(program, &mut size) }, "nvrtcGetProgramLogSize")?;
        if size == 0 { return Ok(String::new()); }
        let mut log = vec![0u8; size];
        check_nvrtc(unsafe { (self.get_log)(program, log.as_mut_ptr().cast::<c_char>()) }, "nvrtcGetProgramLog")?;
        let end = log.iter().position(|byte| *byte == 0).unwrap_or(log.len());
        Ok(String::from_utf8_lossy(&log[..end]).into_owned())
    }
}

/// Módulo PTX cargado en un contexto CUDA.
pub struct CudaKernelModule {
    context: Arc<ContextInner>,
    module: NonNull<c_void>,
    _nvrtc: NvrtcApi,
}

impl CudaKernelModule {
    /// Compila código CUDA C++ con NVRTC y carga el PTX en el dispositivo.
    /// Para una RTX 4060 (Ada Lovelace), usa compute_89.
    pub fn compile(context: &CudaContext, source: &str, program_name: &str, architecture: &str) -> Result<Self> {
        if !valid_architecture(architecture) {
            return Err(Error::new("Arquitectura NVRTC inválida; usa por ejemplo compute_89"));
        }
        context.inner.make_current()?;
        let nvrtc = unsafe { NvrtcApi::load()? };
        let ptx = nvrtc.compile(source, program_name, architecture)?;
        let mut raw_module = std::ptr::null_mut();
        check_cuda(unsafe {
            (context.inner.api.module_load_data)(&mut raw_module, ptx.as_ptr().cast::<c_void>())
        }, "cuModuleLoadData")?;
        let module = NonNull::new(raw_module).ok_or_else(|| Error::new("CUDA devolvió un módulo nulo"))?;
        Ok(Self { context: Arc::clone(&context.inner), module, _nvrtc: nvrtc })
    }

    /// Compila el conjunto inicial de operaciones FP32 de uso general.
    pub fn compile_builtin_ops(context: &CudaContext, architecture: &str) -> Result<Self> {
        Self::compile(context, BUILTIN_KERNELS, "mygpt_ops.cu", architecture)
    }

    /// Suma dos vectores residentes en VRAM: output[i] = a[i] + b[i].
    pub fn elementwise_add(
        &self, a: &CudaBuffer, b: &CudaBuffer, output: &mut CudaBuffer,
        len: usize, stream: Option<&CudaStream>,
    ) -> Result<()> {
        self.validate_buffers(&[a, b, output], stream, len, false)?;
        let len_arg = u64::try_from(len).map_err(|_| Error::new("Longitud CUDA fuera de rango"))?;
        let mut a_arg = a.pointer;
        let mut b_arg = b.pointer;
        let mut out_arg = output.pointer;
        let mut n_arg = len_arg;
        let mut args = [
            (&mut a_arg as *mut u64).cast::<c_void>(),
            (&mut b_arg as *mut u64).cast::<c_void>(),
            (&mut out_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("elementwise_add_f32")?;
        let blocks = blocks_for(len)?;
        unsafe { kernel.launch([blocks, 1, 1], [256, 1, 1], 0, stream, &mut args) }
    }

    /// Multiplica dos vectores residentes en VRAM: output[i] = a[i] * b[i].
    pub fn elementwise_mul(
        &self, a: &CudaBuffer, b: &CudaBuffer, output: &mut CudaBuffer,
        len: usize, stream: Option<&CudaStream>,
    ) -> Result<()> {
        self.validate_buffers(&[a, b, output], stream, len, false)?;
        let mut a_arg = a.pointer;
        let mut b_arg = b.pointer;
        let mut out_arg = output.pointer;
        let mut n_arg = u64::try_from(len).map_err(|_| Error::new("Longitud CUDA fuera de rango"))?;
        let mut args = [
            (&mut a_arg as *mut u64).cast::<c_void>(),
            (&mut b_arg as *mut u64).cast::<c_void>(),
            (&mut out_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("elementwise_mul_f32")?;
        unsafe { kernel.launch([blocks_for(len)?, 1, 1], [256, 1, 1], 0, stream, &mut args) }
    }

    /// Escala un vector residente en VRAM: output[i] = input[i] * factor.
    pub fn scale(
        &self, input: &CudaBuffer, output: &mut CudaBuffer, len: usize,
        factor: f32, stream: Option<&CudaStream>,
    ) -> Result<()> {
        self.validate_buffers(&[input, output], stream, len, false)?;
        let mut input_arg = input.pointer;
        let mut output_arg = output.pointer;
        let mut n_arg = u64::try_from(len).map_err(|_| Error::new("Longitud CUDA fuera de rango"))?;
        let mut factor_arg = factor;
        let mut args = [
            (&mut input_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
            (&mut factor_arg as *mut f32).cast::<c_void>(),
        ];
        let kernel = self.function("scale_f32")?;
        unsafe { kernel.launch([blocks_for(len)?, 1, 1], [256, 1, 1], 0, stream, &mut args) }
    }

    /// Reducción FP32 a un escalar, usando un bloque y memoria compartida.
    pub fn reduce_sum(
        &self, input: &CudaBuffer, output: &mut CudaBuffer, len: usize,
        stream: Option<&CudaStream>,
    ) -> Result<()> {
        if len == 0 { return Err(Error::new("No se puede reducir un vector vacío")); }
        self.validate_buffers(&[input, output], stream, len, true)?;
        let mut input_arg = input.pointer;
        let mut output_arg = output.pointer;
        let mut n_arg = u64::try_from(len).map_err(|_| Error::new("Longitud CUDA fuera de rango"))?;
        let mut args = [
            (&mut input_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("reduce_sum_f32")?;
        unsafe { kernel.launch([1, 1, 1], [256, 1, 1], 256 * 4, stream, &mut args) }
    }

    fn validate_buffers(
        &self, buffers: &[&CudaBuffer], stream: Option<&CudaStream>,
        len: usize, scalar_output: bool,
    ) -> Result<()> {
        if len == 0 { return Err(Error::new("La longitud de la operación CUDA debe ser mayor que cero")); }
        for (index, buffer) in buffers.iter().enumerate() {
            if !Arc::ptr_eq(&self.context, &buffer.context) {
                return Err(Error::new("Todos los buffers deben pertenecer al mismo contexto CUDA"));
            }
            let required = if scalar_output && index == buffers.len() - 1 { 1 } else { len };
            if buffer.len < required {
                return Err(Error::new(format!("Buffer CUDA {index} demasiado pequeño: {} < {required}", buffer.len)));
            }
        }
        if let Some(stream) = stream {
            if !Arc::ptr_eq(&self.context, &stream.context) {
                return Err(Error::new("El stream debe pertenecer al mismo contexto CUDA"));
            }
        }
        Ok(())
    }

    pub fn function<'module>(&'module self, name: &str) -> Result<CudaKernel<'module>> {
        let name = CString::new(name).map_err(|_| Error::new("Nombre de kernel inválido"))?;
        self.context.make_current()?;
        let mut raw_function = std::ptr::null_mut();
        check_cuda(unsafe {
            (self.context.api.module_get_function)(&mut raw_function, self.module.as_ptr(), name.as_ptr())
        }, "cuModuleGetFunction")?;
        let function = NonNull::new(raw_function).ok_or_else(|| Error::new("CUDA devolvió una función de kernel nula"))?;
        Ok(CudaKernel { module: self, function })
    }
}

impl Drop for CudaKernelModule {
    fn drop(&mut self) {
        if self.context.make_current().is_ok() {
            unsafe { (self.context.api.module_unload)(self.module.as_ptr()); }
        }
    }
}

/// Función de kernel ligada a la vida de su módulo.
pub struct CudaKernel<'module> {
    module: &'module CudaKernelModule,
    function: NonNull<c_void>,
}

impl CudaKernel<'_> {
    /// Encola el kernel sin sincronizar.
    ///
    /// # Safety
    /// Debe existir un puntero por parámetro, cada uno debe apuntar a un valor
    /// con el tipo/tamaño ABI esperado, y los buffers deben vivir hasta terminar
    /// el trabajo encolado en el stream.
    pub unsafe fn launch(
        &self, grid: [u32; 3], block: [u32; 3], shared_memory_bytes: u32,
        stream: Option<&CudaStream>, arguments: &mut [*mut c_void],
    ) -> Result<()> {
        if grid.contains(&0) || block.contains(&0) {
            return Err(Error::new("Las dimensiones grid y block deben ser mayores que cero"));
        }
        if let Some(stream) = stream {
            if !Arc::ptr_eq(&self.module.context, &stream.context) {
                return Err(Error::new("El stream y el kernel deben pertenecer al mismo contexto CUDA"));
            }
        }
        self.module.context.make_current()?;
        let raw_stream = stream.map(|s| s.raw.as_ptr()).unwrap_or(std::ptr::null_mut());
        check_cuda((self.module.context.api.launch_kernel)(
            self.function.as_ptr(), grid[0], grid[1], grid[2],
            block[0], block[1], block[2], shared_memory_bytes,
            raw_stream, arguments.as_mut_ptr(), std::ptr::null_mut(),
        ), "cuLaunchKernel")
    }
}

const BUILTIN_KERNELS: &str = r#"
extern "C" __global__ void elementwise_add_f32(
    const float* a, const float* b, float* output, unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] + b[i];
}
extern "C" __global__ void elementwise_mul_f32(
    const float* a, const float* b, float* output, unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] * b[i];
}
extern "C" __global__ void scale_f32(
    const float* input, float* output, unsigned long long n, float factor
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * factor;
}
extern "C" __global__ void reduce_sum_f32(
    const float* input, float* output, unsigned long long n
) {
    extern __shared__ float partial[];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;
    for (unsigned long long i = tid; i < n; i += blockDim.x) sum += input[i];
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[0] = partial[0];
}
"#;

fn blocks_for(len: usize) -> Result<u32> {
    let blocks = len.div_ceil(256);
    u32::try_from(blocks).map_err(|_| Error::new("Grid CUDA fuera de rango"))
}

fn valid_architecture(value: &str) -> bool {
    value.strip_prefix("compute_")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

fn check_nvrtc(status: c_int, operation: &str) -> Result<()> {
    if status == NVRTC_SUCCESS { Ok(()) } else {
        Err(Error::new(format!("NVRTC: {operation} falló con código {status}")))
    }
}
fn check_cuda(status: c_int, operation: &str) -> Result<()> {
    if status == 0 { Ok(()) } else {
        Err(Error::new(format!("CUDA Driver API: {operation} falló con código {status}")))
    }
}

#[cfg(test)]
mod tests {
    use super::valid_architecture;

    #[test]
    fn validates_nvrtc_architecture_without_loading_runtime() {
        assert!(valid_architecture("compute_89"));
        assert!(valid_architecture("compute_120"));
        assert!(!valid_architecture("sm_89"));
        assert!(!valid_architecture("compute_"));
        assert!(!valid_architecture("compute_8x"));
    }
}
