//! Backend tensorial CUDA residente en VRAM.
//!
//! Mantiene shape en host y datos FP32 en un CudaBuffer. Las operaciones
//! elementwise, reducciones, softmax, LayerNorm y matmul no descargan resultados
//! a RAM. La descarga solo ocurre mediante to_host() de forma explícita.

use super::cublas::CudaBlas;
use super::kernels::CudaKernelModule;
use super::{CudaBuffer, CudaContext, CudaRuntime, Error, Result};
use crate::tensor::Tensor;

/// Tensor denso FP32 residente en VRAM.
pub struct CudaTensor {
    shape: Vec<usize>,
    buffer: CudaBuffer,
}

impl CudaTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Backend GPU inicial. El contexto, módulos y handle cuBLAS viven durante
/// toda la vida del backend; los tensores mantienen el contexto vivo también.
pub struct CudaBackend {
    context: CudaContext,
    kernels: CudaKernelModule,
    blas: CudaBlas<'static>,
}

impl CudaBackend {
    /// Detecta arquitectura, crea contexto y compila los kernels integrados.
    pub fn new(runtime: &CudaRuntime, ordinal: usize) -> Result<Self> {
        let device = runtime.device(ordinal)?;
        let architecture = format!("compute_{}{}", device.compute_major, device.compute_minor);
        let context = runtime.create_context(ordinal)?;
        let kernels = CudaKernelModule::compile_builtin_ops(&context, &architecture)?;
        let blas = context.create_blas()?;
        Ok(Self { context, kernels, blas })
    }

    /// Sube un tensor CPU a VRAM. No conserva una copia host.
    pub fn upload(&self, tensor: &Tensor) -> Result<CudaTensor> {
        let shape = tensor.shape().to_vec();
        let count = checked_numel(&shape)?;
        if count != tensor.size() {
            return Err(Error::new("La forma del tensor no coincide con su almacenamiento"));
        }
        let mut buffer = self.context.allocate_f32(count)?;
        buffer.copy_from_host(tensor.data())?;
        Ok(CudaTensor { shape, buffer })
    }

    /// Descarga explícitamente un tensor CUDA al tipo Tensor CPU.
    pub fn to_host(&self, tensor: &CudaTensor) -> Result<Tensor> {
        Tensor::new(tensor.shape.clone(), tensor.buffer.copy_to_host()?)
    }

    /// Reserva un tensor CUDA inicializado a cero.
    pub fn zeros(&self, shape: impl Into<Vec<usize>>) -> Result<CudaTensor> {
        let shape = shape.into();
        let count = checked_numel(&shape)?;
        let mut buffer = self.context.allocate_f32(count)?;
        buffer.copy_from_host(&vec![0.0; count])?;
        Ok(CudaTensor { shape, buffer })
    }

    pub fn add(&self, lhs: &CudaTensor, rhs: &CudaTensor) -> Result<CudaTensor> {
        self.require_same_shape(lhs, rhs)?;
        let mut output = self.allocate_like(lhs)?;
        self.kernels.elementwise_add(&lhs.buffer, &rhs.buffer, &mut output.buffer, lhs.len(), None)?;
        Ok(output)
    }

    pub fn mul(&self, lhs: &CudaTensor, rhs: &CudaTensor) -> Result<CudaTensor> {
        self.require_same_shape(lhs, rhs)?;
        let mut output = self.allocate_like(lhs)?;
        self.kernels.elementwise_mul(&lhs.buffer, &rhs.buffer, &mut output.buffer, lhs.len(), None)?;
        Ok(output)
    }

    pub fn scale(&self, input: &CudaTensor, factor: f32) -> Result<CudaTensor> {
        let mut output = self.allocate_like(input)?;
        self.kernels.scale(&input.buffer, &mut output.buffer, input.len(), factor, None)?;
        Ok(output)
    }

    pub fn sum(&self, input: &CudaTensor) -> Result<CudaTensor> {
        let mut output = self.context.allocate_f32(1)?;
        self.kernels.reduce_sum(&input.buffer, &mut output, input.len(), None)?;
        Ok(CudaTensor { shape: vec![1], buffer: output })
    }

    /// Softmax sobre la última dimensión de un tensor row-major.
    pub fn softmax_last_dim(&self, input: &CudaTensor) -> Result<CudaTensor> {
        let columns = *input.shape.last().ok_or_else(|| Error::new("Softmax requiere un tensor con al menos una dimensión"))?;
        let rows = input.len() / columns;
        let mut output = self.allocate_like(input)?;
        self.kernels.softmax_rows(&input.buffer, &mut output.buffer, rows, columns, None)?;
        Ok(output)
    }

    /// LayerNorm sobre la última dimensión; gamma y beta deben tener shape [columns].
    pub fn layer_norm_last_dim(
        &self, input: &CudaTensor, gamma: &CudaTensor, beta: &CudaTensor, epsilon: f32,
    ) -> Result<CudaTensor> {
        let columns = *input.shape.last().ok_or_else(|| Error::new("LayerNorm requiere un tensor con al menos una dimensión"))?;
        if gamma.shape != [columns] || beta.shape != [columns] {
            return Err(Error::new("Gamma y beta deben tener shape [última_dimensión]"));
        }
        let rows = input.len() / columns;
        let mut output = self.allocate_like(input)?;
        self.kernels.layer_norm_rows(
            &input.buffer, &gamma.buffer, &beta.buffer, &mut output.buffer,
            rows, columns, epsilon, None,
        )?;
        Ok(output)
    }

    /// Producto matricial 2D FP32 con cuBLAS.
    pub fn matmul(&self, lhs: &CudaTensor, rhs: &CudaTensor) -> Result<CudaTensor> {
        if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
            return Err(Error::new("matmul CUDA requiere A=[m,k] y B=[k,n]"));
        }
        let rows = lhs.shape[0];
        let inner = lhs.shape[1];
        let columns = rhs.shape[1];
        let mut output = CudaTensor {
            shape: vec![rows, columns],
            buffer: self.context.allocate_f32(rows.checked_mul(columns)
                .ok_or_else(|| Error::new("Dimensiones de matmul fuera de rango"))?)?,
        };
        self.blas.matmul_row_major(&lhs.buffer, &rhs.buffer, &mut output.buffer, rows, inner, columns)?;
        Ok(output)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.context.synchronize()
    }

    fn allocate_like(&self, tensor: &CudaTensor) -> Result<CudaTensor> {
        Ok(CudaTensor {
            shape: tensor.shape.clone(),
            buffer: self.context.allocate_f32(tensor.len())?,
        })
    }

    fn require_same_shape(&self, lhs: &CudaTensor, rhs: &CudaTensor) -> Result<()> {
        if lhs.shape != rhs.shape {
            return Err(Error::new("Los tensores CUDA deben tener la misma forma"));
        }
        Ok(())
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(Error::new("La forma CUDA debe tener dimensiones no vacías"));
    }
    shape.iter().try_fold(1usize, |total, dimension| {
        total.checked_mul(*dimension).ok_or_else(|| Error::new("Forma CUDA fuera de rango"))
    })
}
