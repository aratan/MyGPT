//! Forward completo GPT sobre CUDA. Los parámetros se copian una vez y quedan en VRAM.

use super::tensor::{CudaBackend, CudaTensor};
use super::transformer::CudaTransformerBlock;
use crate::error::{Error, Result};
use crate::gpt::GPT;
use crate::tensor::Tensor;

/// Modelo GPT para inferencia CUDA (forward); entrenamiento/backward sigue en CPU.
pub struct CudaGPT {
    vocab_size: usize,
    max_seq_len: usize,
    token_embeddings: CudaTensor,
    positional_embeddings: CudaTensor,
    blocks: Vec<CudaTransformerBlock>,
    final_norm_gamma: CudaTensor,
    final_norm_beta: CudaTensor,
    epsilon: f32,
    output_weights: CudaTensor,
}

impl CudaGPT {
    /// Importa los parámetros actuales del GPT CPU a VRAM.
    pub fn from_cpu(backend: &CudaBackend, model: &GPT) -> Result<Self> {
        let embedding = model.embedding();
        let blocks = model.blocks().iter()
            .map(|block| CudaTransformerBlock::from_cpu(backend, block))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vocab_size: embedding.vocab_size(),
            max_seq_len: embedding.max_seq_len(),
            token_embeddings: backend.upload(embedding.token_embeddings())?,
            positional_embeddings: backend.upload(embedding.positional_embeddings())?,
            blocks,
            final_norm_gamma: backend.upload(model.final_norm().gamma())?,
            final_norm_beta: backend.upload(model.final_norm().beta())?,
            epsilon: model.epsilon(),
            output_weights: backend.upload(model.output_weights())?,
        })
    }

    /// Devuelve logits [batch, seq_len, vocab_size]. Solo la lista de IDs entra desde host.
    pub fn forward(&self, backend: &CudaBackend, token_ids: &[u32],
        batch_size: usize, seq_len: usize) -> Result<CudaTensor> {
        let expected = batch_size.checked_mul(seq_len)
            .ok_or_else(|| Error::new("batch_size * seq_len excede usize"))?;
        if batch_size == 0 || seq_len == 0 || expected != token_ids.len() {
            return Err(Error::new("token_ids debe contener batch_size * seq_len elementos"));
        }
        if seq_len > self.max_seq_len {
            return Err(Error::new("La secuencia excede max_seq_len del GPT"));
        }
        if token_ids.iter().any(|&id| id as usize >= self.vocab_size) {
            return Err(Error::new("Token ID fuera del vocabulario"));
        }
        let mut hidden = backend.embedding_lookup_for_batch(
            token_ids, batch_size, seq_len, &self.token_embeddings,
        )?;
        hidden = backend.add_position_embeddings(&hidden, &self.positional_embeddings)?;
        for block in &self.blocks {
            hidden = block.forward(backend, &hidden)?;
        }
        hidden = backend.layer_norm_last_dim(
            &hidden, &self.final_norm_gamma, &self.final_norm_beta, self.epsilon,
        )?;
        backend.linear(&hidden, &self.output_weights)
    }

    /// Descarga explícitamente logits para decodificación host o comparación CPU.
    pub fn logits_to_host(&self, backend: &CudaBackend, logits: &CudaTensor) -> Result<Tensor> {
        backend.to_host(logits)
    }
}
