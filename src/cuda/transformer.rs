//! TransformerBlock pre-norm ejecutado íntegramente en CUDA.

use super::tensor::{CudaBackend, CudaTensor};
use crate::error::{Error, Result};
use crate::transformer::TransformerBlock;

/// Copia los parámetros de un bloque CPU a VRAM y ejecuta forward en CUDA.
/// Los parámetros permanecen residentes durante toda la vida de esta estructura.
pub struct CudaTransformerBlock {
    num_heads: usize,
    epsilon: f32,
    norm1_gamma: CudaTensor,
    norm1_beta: CudaTensor,
    norm2_gamma: CudaTensor,
    norm2_beta: CudaTensor,
    q_weights: CudaTensor,
    k_weights: CudaTensor,
    v_weights: CudaTensor,
    output_weights: CudaTensor,
    fc1_weights: CudaTensor,
    fc2_weights: CudaTensor,
}

impl CudaTransformerBlock {
    /// Sube una única vez todos los parámetros del bloque CPU a VRAM.
    pub fn from_cpu(backend: &CudaBackend, block: &TransformerBlock) -> Result<Self> {
        let attention = block.attention();
        Ok(Self {
            num_heads: block.num_heads(),
            epsilon: block.epsilon(),
            norm1_gamma: backend.upload(block.norm1().gamma())?,
            norm1_beta: backend.upload(block.norm1().beta())?,
            norm2_gamma: backend.upload(block.norm2().gamma())?,
            norm2_beta: backend.upload(block.norm2().beta())?,
            q_weights: backend.upload(attention.q_weights())?,
            k_weights: backend.upload(attention.k_weights())?,
            v_weights: backend.upload(attention.v_weights())?,
            output_weights: backend.upload(attention.output_weights())?,
            fc1_weights: backend.upload(block.fc1_weights())?,
            fc2_weights: backend.upload(block.fc2_weights())?,
        })
    }

    /// Ejecuta LayerNorm -> atención causal -> residual -> LayerNorm -> MLP GELU -> residual.
    pub fn forward(&self, backend: &CudaBackend, input: &CudaTensor) -> Result<CudaTensor> {
        if input.shape().len() != 3 || input.shape()[2] != self.norm1_gamma.shape()[0] {
            return Err(Error::new("CUDA TransformerBlock espera [batch, seq, embedding]"));
        }
        let normalized_attention = backend.layer_norm_last_dim(
            input, &self.norm1_gamma, &self.norm1_beta, self.epsilon,
        )?;
        let attention = backend.multi_head_attention(
            &normalized_attention,
            &self.q_weights, &self.k_weights, &self.v_weights, &self.output_weights,
            self.num_heads,
        )?;
        let residual = backend.add(input, &attention)?;
        let normalized_mlp = backend.layer_norm_last_dim(
            &residual, &self.norm2_gamma, &self.norm2_beta, self.epsilon,
        )?;
        let hidden = backend.linear(&normalized_mlp, &self.fc1_weights)?;
        let activated = backend.gelu(&hidden)?;
        let mlp_output = backend.linear(&activated, &self.fc2_weights)?;
        backend.add(&residual, &mlp_output)
    }
}
