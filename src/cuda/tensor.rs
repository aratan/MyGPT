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

    /// Embedding lookup GPU; los IDs se validan en CPU y la tabla permanece en VRAM.
    pub fn embedding_lookup(&self, ids: &[u32], table: &CudaTensor) -> Result<CudaTensor> {
        if table.shape.len() != 2 {
            return Err(Error::new("La tabla de embeddings debe tener shape [vocab, dim]"));
        }
        let vocab = table.shape[0];
        let dim = table.shape[1];
        let buffer = self.kernels.embedding_lookup(&self.context, ids, &table.buffer, vocab, dim, None)?;
        Ok(CudaTensor { shape: vec![ids.len(), dim], buffer })
    }

    /// Lookup con shape de secuencia [batch, seq_len, embedding_dim].
    pub fn embedding_lookup_batched(
        &self, ids: &[u32], batch_size: usize, seq_len: usize, table: &CudaTensor,
    ) -> Result<CudaTensor> {
        let expected = batch_size.checked_mul(seq_len)
            .ok_or_else(|| Error::new("Dimensiones batch/seq fuera de rango"))?;
        if batch_size == 0 || seq_len == 0 || ids.len() != expected {
            return Err(Error::new("Los IDs no coinciden con batch_size * seq_len"));
        }
        let mut output = self.embedding_lookup(ids, table)?;
        output.shape = vec![batch_size, seq_len, table.shape[1]];
        Ok(output)
    }

    /// Proyección lineal sobre la última dimensión de entradas rank-2 o rank-3.
    /// Los pesos usan layout row-major [in_features, out_features].
    pub fn linear(&self, input: &CudaTensor, weights: &CudaTensor) -> Result<CudaTensor> {
        if !(input.shape.len() == 2 || input.shape.len() == 3) || weights.shape.len() != 2 {
            return Err(Error::new("linear CUDA requiere input rank 2/3 y pesos rank 2"));
        }
        let input_features = *input.shape.last().ok_or_else(|| Error::new("Input vacío"))?;
        if weights.shape[0] != input_features {
            return Err(Error::new("Dimensiones incompatibles en linear CUDA"));
        }
        let rows = input.len() / input_features;
        let output_features = weights.shape[1];
        let output = self.context.allocate_f32(
            rows.checked_mul(output_features).ok_or_else(|| Error::new("Salida linear fuera de rango"))?
        )?;
        let mut result = CudaTensor { shape: vec![rows, output_features], buffer: output };
        self.blas.matmul_row_major(&input.buffer, &weights.buffer, &mut result.buffer, rows, input_features, output_features)?;
        if input.shape.len() == 3 {
            result.shape = vec![input.shape[0], input.shape[1], output_features];
        }
        Ok(result)
    }

    /// Proyección lineal con bias sobre la última dimensión.
    pub fn linear_with_bias(
        &self, input: &CudaTensor, weights: &CudaTensor, bias: &CudaTensor,
    ) -> Result<CudaTensor> {
        if weights.shape.len() != 2 || bias.shape != [weights.shape[1]] {
            return Err(Error::new("Bias debe tener shape [out_features]"));
        }
        let projected = self.linear(input, weights)?;
        let mut output = self.allocate_like(&projected)?;
        let rows = projected.len() / weights.shape[1];
        self.kernels.add_bias_last_dim(
            &projected.buffer, &bias.buffer, &mut output.buffer,
            rows, weights.shape[1], None,
        )?;
        Ok(output)
    }

    /// Añade embeddings posicionales a activaciones [batch, seq_len, dim].
    pub fn add_position_embeddings(
        &self, hidden: &CudaTensor, positions: &CudaTensor,
    ) -> Result<CudaTensor> {
        if hidden.shape.len() != 3 || positions.shape.len() != 2
            || hidden.shape[1] != positions.shape[0]
            || hidden.shape[2] != positions.shape[1] {
            return Err(Error::new(
                "Se requieren activaciones [batch, seq_len, dim] y posiciones [seq_len, dim]",
            ));
        }
        let mut output = self.allocate_like(hidden)?;
        self.kernels.add_position_embeddings(
            &hidden.buffer, &positions.buffer, &mut output.buffer,
            hidden.shape[0], hidden.shape[1], hidden.shape[2], None,
        )?;
        Ok(output)
    }

    /// GELU con la misma aproximación de `erf` que el backend CPU.
    pub fn gelu(&self, input: &CudaTensor) -> Result<CudaTensor> {
        let mut output = self.allocate_like(input)?;
        self.kernels.gelu(&input.buffer, &mut output.buffer, input.len(), None)?;
        Ok(output)
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

    /// Convierte proyecciones [batch, seq, embedding] a [batch, heads, seq, head_dim].
    pub fn split_heads(&self, input: &CudaTensor, heads: usize) -> Result<CudaTensor> {
        if input.shape.len()!=3 || heads==0 || input.shape[2]%heads!=0 {
            return Err(Error::new("split_heads requiere [batch, seq, embedding] divisible por heads"));
        }
        let (batch,seq,embedding)=(input.shape[0],input.shape[1],input.shape[2]);
        let head_dim=embedding/heads;
        let mut output=CudaTensor { shape:vec![batch,heads,seq,head_dim],
            buffer:self.context.allocate_f32(input.len())? };
        self.kernels.reorder_heads(&input.buffer,&mut output.buffer,batch,seq,heads,head_dim,false,None)?;
        Ok(output)
    }

    /// Convierte [batch, heads, seq, head_dim] a [batch, seq, embedding].
    pub fn merge_heads(&self, input: &CudaTensor) -> Result<CudaTensor> {
        if input.shape.len()!=4 {
            return Err(Error::new("merge_heads requiere [batch, heads, seq, head_dim]"));
        }
        let (batch,heads,seq,dim)=(input.shape[0],input.shape[1],input.shape[2],input.shape[3]);
        if batch==0||heads==0||seq==0||dim==0 {
            return Err(Error::new("Dimensiones inválidas en merge_heads"));
        }
        let embedding=heads.checked_mul(dim).ok_or_else(||Error::new("Embedding fuera de rango"))?;
        let mut output=CudaTensor { shape:vec![batch,seq,embedding],
            buffer:self.context.allocate_f32(input.len())? };
        self.kernels.reorder_heads(&input.buffer,&mut output.buffer,batch,seq,heads,dim,true,None)?;
        Ok(output)
    }

    /// Atención multi-cabeza completa a partir de pesos Q/K/V/O residentes en VRAM.
    /// Input [B,S,E], pesos [E,E], salida [B,S,E].
    pub fn multi_head_attention(&self, input:&CudaTensor, q_weights:&CudaTensor,
        k_weights:&CudaTensor, v_weights:&CudaTensor, output_weights:&CudaTensor,
        heads:usize) -> Result<CudaTensor> {
        if input.shape.len()!=3 || heads==0 {
            return Err(Error::new("multi_head_attention requiere input [batch,seq,embedding] y heads > 0"));
        }
        let embedding=input.shape[2];
        if embedding%heads!=0 {
            return Err(Error::new("embedding debe ser divisible por heads"));
        }
        for weights in [q_weights,k_weights,v_weights,output_weights] {
            if weights.shape!=[embedding,embedding] {
                return Err(Error::new("Los pesos Q/K/V/O deben tener shape [embedding,embedding]"));
            }
        }
        let q=self.split_heads(&self.linear(input,q_weights)?,heads)?;
        let k=self.split_heads(&self.linear(input,k_weights)?,heads)?;
        let v=self.split_heads(&self.linear(input,v_weights)?,heads)?;
        let attended=self.causal_attention(&q,&k,&v)?;
        let merged=self.merge_heads(&attended)?;
        self.linear(&merged,output_weights)
    }

    /// Atención causal; Q/K/V tienen shape [batch, heads, seq_len, head_dim].
    pub fn causal_attention(&self, queries: &CudaTensor, keys: &CudaTensor, values: &CudaTensor) -> Result<CudaTensor> {
        if queries.shape.len()!=4 || queries.shape!=keys.shape || queries.shape!=values.shape {
            return Err(Error::new("Q, K y V deben compartir shape [batch, heads, seq_len, head_dim]"));
        }
        let [batch,heads,seq_len,head_dim]=[queries.shape[0],queries.shape[1],queries.shape[2],queries.shape[3]];
        if batch==0 || heads==0 || seq_len==0 || head_dim==0 {
            return Err(Error::new("Las dimensiones de atención deben ser mayores que cero"));
        }
        let mut output=self.allocate_like(queries)?;
        self.kernels.causal_attention(&queries.buffer,&keys.buffer,&values.buffer,&mut output.buffer,
            batch,heads,seq_len,head_dim,None)?;
        Ok(output)
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

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn causal_attention_matches_cpu_reference_when_cuda_is_available() {
        let runtime = match CudaRuntime::discover() {
            Ok(runtime) => runtime,
            Err(_) => return,
        };
        let backend = CudaBackend::new(&runtime, 0).expect("CUDA backend");
        let (batch, heads, seq, dim) = (1usize, 2usize, 3usize, 2usize);
        let count = batch * heads * seq * dim;
        let q_data: Vec<f32> = (0..count).map(|i| (i as f32 - 3.0) * 0.13).collect();
        let k_data: Vec<f32> = (0..count).map(|i| (i as f32 - 2.0) * -0.07).collect();
        let v_data: Vec<f32> = (0..count).map(|i| (i as f32 + 1.0) * 0.11).collect();
        let shape = vec![batch, heads, seq, dim];
        let q = backend.upload(&Tensor::new(shape.clone(), q_data.clone()).unwrap()).unwrap();
        let k = backend.upload(&Tensor::new(shape.clone(), k_data.clone()).unwrap()).unwrap();
        let v = backend.upload(&Tensor::new(shape.clone(), v_data.clone()).unwrap()).unwrap();
        let actual = backend.to_host(&backend.causal_attention(&q, &k, &v).unwrap()).unwrap();
        backend.synchronize().unwrap();

        let mut expected = vec![0.0f32; count];
        let scale = 1.0 / (dim as f32).sqrt();
        for b in 0..batch {
            for h in 0..heads {
                for query in 0..seq {
                    let base = ((b * heads + h) * seq) * dim;
                    let mut scores = Vec::with_capacity(query + 1);
                    for key in 0..=query {
                        let mut score = 0.0;
                        for d in 0..dim {
                            score += q_data[base + query * dim + d] * k_data[base + key * dim + d];
                        }
                        scores.push(score * scale);
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denominator: f32 = scores.iter().map(|score| (score - max).exp()).sum();
                    for d in 0..dim {
                        let mut value = 0.0;
                        for key in 0..=query {
                            let weight = (scores[key] - max).exp() / denominator;
                            value += weight * v_data[base + key * dim + d];
                        }
                        expected[base + query * dim + d] = value;
                    }
                }
            }
        }
        for (got, want) in actual.data().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 2.0e-4, "CUDA {got} != CPU {want}");
        }
    }
}
