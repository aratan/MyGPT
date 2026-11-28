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

    /// Busca filas de una tabla de embeddings usando IDs u32.
    pub fn embedding_lookup(
        &self, context: &CudaContext, ids: &[u32], table: &CudaBuffer, vocab_size: usize,
        embedding_dim: usize, stream: Option<&CudaStream>,
    ) -> Result<CudaBuffer> {
        if !Arc::ptr_eq(&self.context, &context.inner) {
            return Err(Error::new("El contexto no coincide con el módulo CUDA"));
        }
        if ids.is_empty() || vocab_size == 0 || embedding_dim == 0
            || table.len < vocab_size.checked_mul(embedding_dim)
                .ok_or_else(|| Error::new("Tabla de embeddings fuera de rango"))? {
            return Err(Error::new("Dimensiones inválidas para embedding lookup"));
        }
        if ids.iter().any(|id| *id as usize >= vocab_size) {
            return Err(Error::new("Token ID fuera del vocabulario"));
        }
        if !Arc::ptr_eq(&self.context, &table.context) {
            return Err(Error::new("La tabla de embeddings pertenece a otro contexto CUDA"));
        }
        if let Some(stream) = stream {
            if !Arc::ptr_eq(&self.context, &stream.context) {
                return Err(Error::new("El stream pertenece a otro contexto CUDA"));
            }
        }
        let mut id_buffer = context.allocate_f32(ids.len())?;
        let encoded: Vec<f32> = ids.iter().map(|id| f32::from_bits(*id)).collect();
        id_buffer.copy_from_host(&encoded)?;
        let output_len = ids.len().checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("Output embeddings fuera de rango"))?;
        let mut output = context.allocate_f32(output_len)?;
        let mut ids_ptr = id_buffer.pointer;
        let mut table_ptr = table.pointer;
        let mut output_ptr = output.pointer;
        let mut token_count = ids.len() as u64;
        let mut vocab = vocab_size as u64;
        let mut dim = embedding_dim as u64;
        let mut args = [
            (&mut ids_ptr as *mut u64).cast::<c_void>(),
            (&mut table_ptr as *mut u64).cast::<c_void>(),
            (&mut output_ptr as *mut u64).cast::<c_void>(),
            (&mut token_count as *mut u64).cast::<c_void>(),
            (&mut vocab as *mut u64).cast::<c_void>(),
            (&mut dim as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("embedding_lookup_f32")?;
        unsafe { kernel.launch([blocks_for(output_len)?, 1, 1], [256, 1, 1], 0, stream, &mut args)?; }
        // IDs deben permanecer vivos hasta que el kernel haya consumido el buffer.
        // La sincronización local evita liberar el buffer prematuramente.
        if let Some(stream) = stream { stream.synchronize()?; } else { context.synchronize()?; }
        Ok(output)
    }

    /// Suma un vector bias a cada fila de una matriz row-major [rows, columns].
    pub fn add_bias_last_dim(
        &self, input: &CudaBuffer, bias: &CudaBuffer, output: &mut CudaBuffer,
        rows: usize, columns: usize, stream: Option<&CudaStream>,
    ) -> Result<()> {
        let total = rows.checked_mul(columns)
            .ok_or_else(|| Error::new("Dimensiones bias CUDA fuera de rango"))?;
        if rows == 0 || columns == 0 || input.len < total || output.len < total || bias.len < columns {
            return Err(Error::new("Buffers o dimensiones incompatibles en bias CUDA"));
        }
        self.validate_contexts(&[input, bias, output], stream)?;
        let mut input_arg = input.pointer;
        let mut bias_arg = bias.pointer;
        let mut output_arg = output.pointer;
        let mut rows_arg = u64::try_from(rows).map_err(|_| Error::new("Filas bias fuera de rango"))?;
        let mut columns_arg = u64::try_from(columns).map_err(|_| Error::new("Columnas bias fuera de rango"))?;
        let mut args = [
            (&mut input_arg as *mut u64).cast::<c_void>(),
            (&mut bias_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut columns_arg as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("add_bias_last_dim_f32")?;
        unsafe { kernel.launch([blocks_for(total)?, 1, 1], [256, 1, 1], 0, stream, &mut args) }
    }

    /// Añade embeddings posicionales [seq_len, dim] a activaciones [batch, seq_len, dim].
    pub fn add_position_embeddings(
        &self, hidden: &CudaBuffer, positions: &CudaBuffer, output: &mut CudaBuffer,
        batch_size: usize, seq_len: usize, embedding_dim: usize,
        stream: Option<&CudaStream>,
    ) -> Result<()> {
        let token_count = batch_size.checked_mul(seq_len)
            .ok_or_else(|| Error::new("Dimensiones batch/seq fuera de rango"))?;
        let total = token_count.checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("Dimensiones de embeddings posicionales fuera de rango"))?;
        let position_len = seq_len.checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("Tabla posicional fuera de rango"))?;
        if batch_size == 0 || seq_len == 0 || embedding_dim == 0
            || hidden.len < total || output.len < total || positions.len < position_len {
            return Err(Error::new("Buffers o dimensiones incompatibles en embeddings posicionales CUDA"));
        }
        self.validate_contexts(&[hidden, positions, output], stream)?;
        let mut hidden_arg = hidden.pointer;
        let mut positions_arg = positions.pointer;
        let mut output_arg = output.pointer;
        let mut token_count_arg = u64::try_from(token_count)
            .map_err(|_| Error::new("Número de tokens fuera de rango"))?;
        let mut seq_len_arg = u64::try_from(seq_len)
            .map_err(|_| Error::new("Longitud de secuencia fuera de rango"))?;
        let mut dim_arg = u64::try_from(embedding_dim)
            .map_err(|_| Error::new("Dimensión embedding fuera de rango"))?;
        let mut args = [
            (&mut hidden_arg as *mut u64).cast::<c_void>(),
            (&mut positions_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut token_count_arg as *mut u64).cast::<c_void>(),
            (&mut seq_len_arg as *mut u64).cast::<c_void>(),
            (&mut dim_arg as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("add_position_embeddings_f32")?;
        unsafe { kernel.launch([blocks_for(total)?, 1, 1], [256, 1, 1], 0, stream, &mut args) }
    }

    /// GELU aproximada usada habitualmente en GPT, aplicada elemento a elemento.
    pub fn gelu(
        &self, input: &CudaBuffer, output: &mut CudaBuffer,
        len: usize, stream: Option<&CudaStream>,
    ) -> Result<()> {
        self.validate_buffers(&[input, output], stream, len, false)?;
        let mut input_arg = input.pointer;
        let mut output_arg = output.pointer;
        let mut n_arg = u64::try_from(len).map_err(|_| Error::new("Longitud GELU fuera de rango"))?;
        let mut args = [
            (&mut input_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut n_arg as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("gelu_erf_f32")?;
        unsafe { kernel.launch([blocks_for(len)?, 1, 1], [256, 1, 1], 0, stream, &mut args) }
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

    /// Softmax estable por fila para una matriz row-major [rows, columns].
    /// Atención causal sobre Q/K/V [batch, heads, seq, head_dim].
    pub fn causal_attention(&self, queries: &CudaBuffer, keys: &CudaBuffer, values: &CudaBuffer,
        output: &mut CudaBuffer, batch: usize, heads: usize, seq: usize, head_dim: usize,
        stream: Option<&CudaStream>) -> Result<()> {
        let total = batch.checked_mul(heads).and_then(|n| n.checked_mul(seq))
            .and_then(|n| n.checked_mul(head_dim)).ok_or_else(|| Error::new("Dimensiones de atención fuera de rango"))?;
        let rows = batch.checked_mul(heads).and_then(|n| n.checked_mul(seq))
            .ok_or_else(|| Error::new("Filas de atención fuera de rango"))?;
        if batch == 0 || heads == 0 || seq == 0 || head_dim == 0
            || queries.len < total || keys.len < total || values.len < total || output.len < total {
            return Err(Error::new("Buffers o dimensiones incompatibles en atención causal CUDA"));
        }
        self.validate_contexts(&[queries, keys, values, output], stream)?;
        let mut q=queries.pointer; let mut k=keys.pointer; let mut v=values.pointer; let mut o=output.pointer;
        let mut b=batch as u64; let mut h=heads as u64; let mut s=seq as u64; let mut d=head_dim as u64;
        let mut args=[
            (&mut q as *mut u64).cast::<c_void>(), (&mut k as *mut u64).cast::<c_void>(),
            (&mut v as *mut u64).cast::<c_void>(), (&mut o as *mut u64).cast::<c_void>(),
            (&mut b as *mut u64).cast::<c_void>(), (&mut h as *mut u64).cast::<c_void>(),
            (&mut s as *mut u64).cast::<c_void>(), (&mut d as *mut u64).cast::<c_void>(),
        ];
        let kernel=self.function("causal_attention_f32")?;
        unsafe { kernel.launch([blocks_for(rows)?,1,1],[256,1,1],0,stream,&mut args) }
    }

    pub fn softmax_rows(
        &self, input: &CudaBuffer, output: &mut CudaBuffer,
        rows: usize, columns: usize, stream: Option<&CudaStream>,
    ) -> Result<()> {
        let total = rows.checked_mul(columns).ok_or_else(|| Error::new("Dimensiones softmax fuera de rango"))?;
        if rows == 0 || columns == 0 || input.len < total || output.len < total {
            return Err(Error::new("Buffers o dimensiones incompatibles en softmax CUDA"));
        }
        self.validate_contexts(&[input, output], stream)?;
        let mut input_arg = input.pointer;
        let mut output_arg = output.pointer;
        let mut rows_arg = u64::try_from(rows).map_err(|_| Error::new("Filas fuera de rango"))?;
        let mut columns_arg = u64::try_from(columns).map_err(|_| Error::new("Columnas fuera de rango"))?;
        let mut args = [
            (&mut input_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut columns_arg as *mut u64).cast::<c_void>(),
        ];
        let kernel = self.function("softmax_rows_f32")?;
        unsafe { kernel.launch([u32::try_from(rows).map_err(|_| Error::new("Grid softmax fuera de rango"))?, 1, 1],
            [256, 1, 1], 256 * 2 * 4, stream, &mut args) }
    }

    /// LayerNorm por fila: (x - mean) / sqrt(var + epsilon) * gamma + beta.
    pub fn layer_norm_rows(
        &self, input: &CudaBuffer, gamma: &CudaBuffer, beta: &CudaBuffer,
        output: &mut CudaBuffer, rows: usize, columns: usize, epsilon: f32,
        stream: Option<&CudaStream>,
    ) -> Result<()> {
        let total = rows.checked_mul(columns).ok_or_else(|| Error::new("Dimensiones LayerNorm fuera de rango"))?;
        if rows == 0 || columns == 0 || input.len < total || output.len < total
            || gamma.len < columns || beta.len < columns || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::new("Buffers, dimensiones o epsilon incompatibles en LayerNorm CUDA"));
        }
        self.validate_contexts(&[input, gamma, beta, output], stream)?;
        let mut input_arg = input.pointer;
        let mut gamma_arg = gamma.pointer;
        let mut beta_arg = beta.pointer;
        let mut output_arg = output.pointer;
        let mut rows_arg = u64::try_from(rows).map_err(|_| Error::new("Filas fuera de rango"))?;
        let mut columns_arg = u64::try_from(columns).map_err(|_| Error::new("Columnas fuera de rango"))?;
        let mut epsilon_arg = epsilon;
        let mut args = [
            (&mut input_arg as *mut u64).cast::<c_void>(),
            (&mut gamma_arg as *mut u64).cast::<c_void>(),
            (&mut beta_arg as *mut u64).cast::<c_void>(),
            (&mut output_arg as *mut u64).cast::<c_void>(),
            (&mut rows_arg as *mut u64).cast::<c_void>(),
            (&mut columns_arg as *mut u64).cast::<c_void>(),
            (&mut epsilon_arg as *mut f32).cast::<c_void>(),
        ];
        let kernel = self.function("layer_norm_rows_f32")?;
        unsafe { kernel.launch([u32::try_from(rows).map_err(|_| Error::new("Grid LayerNorm fuera de rango"))?, 1, 1],
            [256, 1, 1], 256 * 2 * 4, stream, &mut args) }
    }

    fn validate_contexts(&self, buffers: &[&CudaBuffer], stream: Option<&CudaStream>) -> Result<()> {
        for buffer in buffers {
            if !Arc::ptr_eq(&self.context, &buffer.context) {
                return Err(Error::new("Todos los buffers deben pertenecer al mismo contexto CUDA"));
            }
        }
        if let Some(stream) = stream {
            if !Arc::ptr_eq(&self.context, &stream.context) {
                return Err(Error::new("El stream debe pertenecer al mismo contexto CUDA"));
            }
        }
        Ok(())
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
extern "C" __global__ void embedding_lookup_f32(
    const unsigned int* token_ids, const float* table, float* output,
    unsigned long long token_count, unsigned long long vocab_size,
    unsigned long long embedding_dim
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = token_count * embedding_dim;
    if (index >= total) return;
    unsigned long long token_index = index / embedding_dim;
    unsigned long long column = index % embedding_dim;
    unsigned int token = token_ids[token_index];
    output[index] = token < vocab_size ? table[(unsigned long long)token * embedding_dim + column] : 0.0f;
}
extern "C" __global__ void add_position_embeddings_f32(
    float* hidden, const float* positions, unsigned long long token_count,
    unsigned long long seq_len, unsigned long long embedding_dim
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = token_count * embedding_dim;
    if (index >= total) return;
    unsigned long long token_index = index / embedding_dim;
    unsigned long long column = index % embedding_dim;
    unsigned long long position = token_index % seq_len;
    hidden[index] += positions[position * embedding_dim + column];
}
extern "C" __global__ void add_bias_last_dim_f32(
    const float* input, const float* bias, float* output,
    unsigned long long rows, unsigned long long columns
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = rows * columns;
    if (index >= total) return;
    output[index] = input[index] + bias[index % columns];
}
extern "C" __global__ void gelu_erf_f32(
    const float* input, float* output, unsigned long long n
) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = input[i];
    const float z = fabsf(x) * 0.7071067811865475f;
    const float t = 1.0f / (1.0f + 0.3275911f * z);
    const float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    const float erf_abs = 1.0f - polynomial * expf(-z * z);
    const float erf_value = x < 0.0f ? -erf_abs : erf_abs;
    output[i] = 0.5f * x * (1.0f + erf_value);
}
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
extern "C" __global__ void causal_attention_f32(
 const float* q,const float* k,const float* v,float* output,
 unsigned long long batch,unsigned long long heads,unsigned long long seq,unsigned long long head_dim) {
 unsigned long long row=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=batch*heads*seq)return;
 unsigned long long query=row%seq, base=(row/seq)*seq*head_dim, qo=base+query*head_dim;
 float scale=rsqrtf((float)head_dim), maximum=-INFINITY;
 for(unsigned long long key=0;key<=query;++key){
  unsigned long long ko=base+key*head_dim; float score=0.0f;
  for(unsigned long long d=0;d<head_dim;++d)score+=q[qo+d]*k[ko+d];
  maximum=fmaxf(maximum,score*scale);
 }
 float denominator=0.0f;
 for(unsigned long long key=0;key<=query;++key){
  unsigned long long ko=base+key*head_dim; float score=0.0f;
  for(unsigned long long d=0;d<head_dim;++d)score+=q[qo+d]*k[ko+d];
  denominator+=expf(score*scale-maximum);
 }
 for(unsigned long long d=0;d<head_dim;++d){
  float acc=0.0f;
  for(unsigned long long key=0;key<=query;++key){
   unsigned long long ko=base+key*head_dim; float score=0.0f;
   for(unsigned long long j=0;j<head_dim;++j)score+=q[qo+j]*k[ko+j];
   acc+=(expf(score*scale-maximum)/denominator)*v[ko+d];
  }
  output[qo+d]=acc;
 }
}
extern "C" __global__ void softmax_rows_f32(
    const float* input, float* output, unsigned long long rows, unsigned long long columns
) {
    unsigned long long row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float* maxima = shared;
    float* sums = shared + blockDim.x;
    unsigned int tid = threadIdx.x;
    float local_max = -INFINITY;
    for (unsigned long long col = tid; col < columns; col += blockDim.x)
        local_max = fmaxf(local_max, input[row * columns + col]);
    maxima[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
        __syncthreads();
    }
    float row_max = maxima[0];
    float local_sum = 0.0f;
    for (unsigned long long col = tid; col < columns; col += blockDim.x) {
        float value = expf(input[row * columns + col] - row_max);
        output[row * columns + col] = value;
        local_sum += value;
    }
    sums[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    float denominator = sums[0];
    for (unsigned long long col = tid; col < columns; col += blockDim.x)
        output[row * columns + col] /= denominator;
}
extern "C" __global__ void layer_norm_rows_f32(
    const float* input, const float* gamma, const float* beta, float* output,
    unsigned long long rows, unsigned long long columns, float epsilon
) {
    unsigned long long row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float* sums = shared;
    float* squares = shared + blockDim.x;
    unsigned int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (unsigned long long col = tid; col < columns; col += blockDim.x)
        local_sum += input[row * columns + col];
    sums[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    float mean = sums[0] / (float)columns;
    float local_variance = 0.0f;
    for (unsigned long long col = tid; col < columns; col += blockDim.x) {
        float centered = input[row * columns + col] - mean;
        local_variance += centered * centered;
    }
    squares[tid] = local_variance;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) squares[tid] += squares[tid + stride];
        __syncthreads();
    }
    float variance = squares[0] / (float)columns;
    float inverse_std = rsqrtf(variance + epsilon);
    for (unsigned long long col = tid; col < columns; col += blockDim.x) {
        float normalized = (input[row * columns + col] - mean) * inverse_std;
        output[row * columns + col] = normalized * gamma[col] + beta[col];
    }
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
