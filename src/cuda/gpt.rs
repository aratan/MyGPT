//! Forward completo GPT sobre CUDA. Los parámetros se copian una vez y quedan en VRAM.

use super::tensor::{CudaBackend, CudaTensor};
use super::transformer::CudaTransformerBlock;
use crate::error::{Error, Result};
use crate::gpt::{GenerationConfig, GPT, SamplingStrategy, StopCondition};
use crate::rng::Rng;
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
        let mut hidden = backend.embedding_lookup_batched(
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

    /// Genera tokens usando el forward CUDA y el sampler común del modelo.
    ///
    /// Esta primera versión descarga los logits de cada paso para seleccionar el
    /// token en CPU. No implementa KV cache: vuelve a evaluar el contexto completo.
    /// Así conserva exactamente las reglas de sampling existentes mientras se
    /// desarrolla la decodificación y caché residentes en GPU.
    pub fn generate(
        &self,
        backend: &CudaBackend,
        prompt: &[u32],
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::new("Generation requires a non-empty prompt"));
        }
        if prompt.iter().any(|&id| id as usize >= self.vocab_size) {
            return Err(Error::new("Prompt token ID is outside the model vocabulary"));
        }
        if config.max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        let mut context = prompt.to_vec();
        let mut generated = Vec::with_capacity(config.max_new_tokens);
        let mut rng = Rng::new(config.seed);
        for _ in 0..config.max_new_tokens {
            if context.len() > self.max_seq_len {
                let excess = context.len() - self.max_seq_len;
                context.drain(..excess);
            }
            let logits = self.forward(backend, &context, 1, context.len())?;
            let host_logits = backend.to_host(&logits)?;
            let start = (context.len() - 1) * self.vocab_size;
            let mut last_logits =
                host_logits.data()[start..start + self.vocab_size].to_vec();

            if config.repetition_penalty != 1.0 {
                GPT::apply_repetition_penalty(
                    &context,
                    &mut last_logits,
                    config.repetition_penalty,
                    config.repetition_window,
                )?;
            }
            let next = GPT::select_token(
                &last_logits,
                config.temperature,
                config.strategy,
                &mut rng,
                config.top_k,
            )?;
            context.push(next);
            generated.push(next);
            let stop_id = match config.stop_condition {
                StopCondition::EndOfTurn => crate::tokenizer::EOT_ID,
                StopCondition::EndOfConversation => crate::tokenizer::END_ID,
            };
            if next == stop_id {
                break;
            }
        }
        Ok(generated)
    }

    /// Descarga explícitamente logits para decodificación host o comparación CPU.
    pub fn logits_to_host(&self, backend: &CudaBackend, logits: &CudaTensor) -> Result<Tensor> {
        backend.to_host(logits)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::GPT;

    #[test]
    fn complete_gpt_forward_matches_cpu_when_cuda_is_available() {
        let runtime = match super::super::CudaRuntime::discover() {
            Ok(runtime) => runtime,
            Err(_) => return,
        };
        let backend = CudaBackend::new(&runtime, 0).expect("CUDA backend");
        let model = GPT::new(32, 8, 2, 2, 16, 8, 2026, 1.0e-5).unwrap();
        let token_ids = [1u32, 4, 7, 2, 9];
        let expected = model.forward(&token_ids, 1, token_ids.len()).unwrap();
        let gpu_model = CudaGPT::from_cpu(&backend, &model).unwrap();
        let logits = gpu_model.forward(&backend, &token_ids, 1, token_ids.len()).unwrap();
        let actual = gpu_model.logits_to_host(&backend, &logits).unwrap();
        backend.synchronize().unwrap();

        assert_eq!(actual.shape(), expected.shape());
        for (got, want) in actual.data().iter().zip(expected.data()) {
            assert!((got - want).abs() < 2.0e-3, "CUDA {got} != CPU {want}");
        }
    }

    #[test]
    fn complete_gpt_forward_rejects_invalid_sequence_length_when_cuda_is_available() {
        let runtime = match super::super::CudaRuntime::discover() {
            Ok(runtime) => runtime,
            Err(_) => return,
        };
        let backend = CudaBackend::new(&runtime, 0).unwrap();
        let model = GPT::new(16, 8, 2, 1, 16, 4, 9, 1.0e-5).unwrap();
        let gpu_model = CudaGPT::from_cpu(&backend, &model).unwrap();
        assert!(gpu_model.forward(&backend, &[1, 2, 3, 4, 5], 1, 5).is_err());
    }
}
