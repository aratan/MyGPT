//! Modelo GPT completo: forward, backward, generación y checkpoints.

use crate::backprop::{
    accumulate_block, accumulate_layer_norm, embedding_backward, layer_norm_backward,
    linear_backward, transformer_block_backward, GPTGradients,
};
use crate::embedding::Embedding;
use crate::error::{Error, Result};
use crate::layer_norm::LayerNorm;
use crate::loss::{cross_entropy, cross_entropy_backward};
use crate::ops::linear;
use crate::rng::Rng;
use crate::serialize::{read_file, BytesWriter};
use crate::tensor::Tensor;
use crate::transformer::TransformerBlock;
use std::path::Path;

const INITIALIZATION_SCALE: f32 = 0.02;
const FILE_MAGIC: &[u8; 8] = b"MINIGPTM";
const FILE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingStrategy {
    Greedy,
    Sampling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopCondition {
    EndOfTurn,
    EndOfConversation,
}

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub strategy: SamplingStrategy,
    pub stop_condition: StopCondition,
    pub seed: u32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 16,
            temperature: 1.0,
            strategy: SamplingStrategy::Greedy,
            stop_condition: StopCondition::EndOfConversation,
            seed: 42,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_window: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GPTParameterCount {
    pub embeddings: usize,
    pub transformer_blocks: usize,
    pub final_layer_norm: usize,
    pub output_projection: usize,
}

impl GPTParameterCount {
    pub fn total(&self) -> usize {
        self.embeddings + self.transformer_blocks + self.final_layer_norm + self.output_projection
    }
}

/// Préstamo mutable de un parámetro y su gradiente, para el optimizador.
pub struct ParamBinding<'a> {
    pub parameter: &'a mut Tensor,
    pub gradient: &'a Tensor,
    pub weight_decay: bool,
    pub name: String,
}

/// Préstamo inmutable de un parámetro y su gradiente.
pub struct ParamRef<'a> {
    pub parameter: &'a Tensor,
    pub gradient: &'a Tensor,
    pub weight_decay: bool,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct GPT {
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    output_weights: Tensor,
    gradients: GPTGradients,
}

impl GPT {
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        num_heads: usize,
        num_layers: usize,
        mlp_hidden_dim: usize,
        max_seq_len: usize,
        seed: u32,
        epsilon: f32,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(Error::new("num_layers must be greater than zero"));
        }
        let embedding = Embedding::new(vocab_size, embedding_dim, max_seq_len, seed)?;
        let final_norm = LayerNorm::new(embedding_dim, epsilon)?;
        let output_weights = Tensor::random(
            [embedding_dim, vocab_size],
            component_seed(seed, num_layers),
            -INITIALIZATION_SCALE,
            INITIALIZATION_SCALE,
        )?;
        let mut blocks = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            blocks.push(TransformerBlock::new(
                embedding_dim,
                num_heads,
                mlp_hidden_dim,
                component_seed(seed, layer),
                epsilon,
            )?);
        }
        let gradients = GPTGradients::zeros(&embedding, &blocks, &final_norm, &output_weights);
        let model = Self {
            embedding,
            blocks,
            final_norm,
            output_weights,
            gradients,
        };
        model.validate_parameters()?;
        Ok(model)
    }

    pub fn forward(&self, token_ids: &[u32], batch_size: usize, seq_len: usize) -> Result<Tensor> {
        self.validate_parameters()?;
        let features = self.forward_features(token_ids, batch_size, seq_len)?;
        linear(&features, &self.output_weights)
    }

    pub fn forward_features(
        &self,
        token_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let mut hidden = self.embedding.forward(token_ids, batch_size, seq_len)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.final_norm.forward(&hidden)
    }

    /// Calcula pérdida y acumula gradientes (no los pone a cero).
    pub fn backward(
        &mut self,
        token_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
        targets: &[u32],
        mask: &[u8],
    ) -> Result<f32> {
        self.validate_parameters()?;
        let mut block_inputs = Vec::with_capacity(self.blocks.len());
        let mut hidden = self.embedding.forward(token_ids, batch_size, seq_len)?;
        for block in &self.blocks {
            block_inputs.push(hidden);
            hidden = block.forward(&block_inputs[block_inputs.len() - 1])?;
        }

        let final_input = hidden;
        let features = self.final_norm.forward(&final_input)?;
        let logits = linear(&features, &self.output_weights)?;
        let loss = cross_entropy(&logits, targets, mask)?;
        let logits_gradient = cross_entropy_backward(&logits, targets, mask)?;

        let output_backward = linear_backward(&features, &self.output_weights, &logits_gradient)?;
        self.gradients
            .output_weights
            .add_assign(&output_backward.weight_gradient)?;
        let final_norm_backward = layer_norm_backward(
            &self.final_norm,
            &final_input,
            &output_backward.input_gradient,
        )?;
        accumulate_layer_norm(&mut self.gradients.final_norm, &final_norm_backward.parameters)?;
        let mut hidden_gradient = final_norm_backward.input_gradient;

        for layer in (0..self.blocks.len()).rev() {
            let block_backward =
                transformer_block_backward(&self.blocks[layer], &block_inputs[layer], &hidden_gradient)?;
            accumulate_block(&mut self.gradients.blocks[layer], &block_backward.parameters)?;
            hidden_gradient = block_backward.input_gradient;
        }

        let embedding_gradients =
            embedding_backward(&self.embedding, token_ids, batch_size, seq_len, &hidden_gradient)?;
        self.gradients
            .embedding
            .token_embeddings
            .add_assign(&embedding_gradients.token_embeddings)?;
        self.gradients
            .embedding
            .positional_embeddings
            .add_assign(&embedding_gradients.positional_embeddings)?;
        Ok(loss)
    }

    /// Devuelve solo los tokens nuevos, incluido el especial que detiene la generación.
    pub fn generate(&self, prompt: &[u32], config: &GenerationConfig) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::new("Generation requires a non-empty prompt"));
        }
        for &id in prompt {
            if id as usize >= self.vocab_size() {
                return Err(Error::new(
                    "Prompt token ID is outside the model vocabulary",
                ));
            }
        }
        let mut generated = Vec::with_capacity(config.max_new_tokens);
        if config.max_new_tokens == 0 {
            return Ok(generated);
        }

        let mut context = prompt.to_vec();
        let mut rng = Rng::new(config.seed);
        for _ in 0..config.max_new_tokens {
            if context.len() > self.max_seq_len() {
                let excess = context.len() - self.max_seq_len();
                context.drain(..excess);
            }
            let logits = self.forward(&context, 1, context.len())?;
            let vocabulary = self.vocab_size();
            let start = (context.len() - 1) * vocabulary;
            let last_logits = &logits.data()[start..start + vocabulary];

            let adjusted = if config.repetition_penalty != 1.0 {
                let mut adjusted = last_logits.to_vec();
                Self::apply_repetition_penalty(
                    &context,
                    &mut adjusted,
                    config.repetition_penalty,
                    config.repetition_window,
                )?;
                adjusted
            } else {
                last_logits.to_vec()
            };
            let next = Self::select_token(
                &adjusted,
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

    pub fn softmax(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
        if logits.is_empty() {
            return Err(Error::new("softmax requires at least one logit"));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Error::new(
                "Temperature must be finite and greater than zero",
            ));
        }
        let mut maximum = f32::NEG_INFINITY;
        for &value in logits {
            if !value.is_finite() {
                return Err(Error::new("Logits must be finite"));
            }
            maximum = maximum.max(value);
        }
        let mut probabilities = vec![0.0f32; logits.len()];
        let mut denominator = 0.0f64;
        for (index, &value) in logits.iter().enumerate() {
            let weight = ((f64::from(value) - f64::from(maximum)) / f64::from(temperature)).exp();
            probabilities[index] = weight as f32;
            denominator += weight;
        }
        for probability in &mut probabilities {
            *probability = (f64::from(*probability) / denominator) as f32;
        }
        Ok(probabilities)
    }

    pub fn select_token(
        logits: &[f32],
        temperature: f32,
        strategy: SamplingStrategy,
        rng: &mut Rng,
        top_k: usize,
    ) -> Result<u32> {
        if logits.is_empty() {
            return Err(Error::new("Token selection requires at least one logit"));
        }
        if logits.len() > u32::MAX as usize {
            return Err(Error::new("Logit count exceeds TokenId range"));
        }
        if top_k > logits.len() {
            return Err(Error::new("top_k exceeds vocabulary size"));
        }
        if strategy == SamplingStrategy::Greedy {
            let mut best = 0usize;
            for index in 1..logits.len() {
                if logits[index] > logits[best] {
                    best = index;
                }
            }
            return Ok(best as u32);
        }

        let mut candidates: Vec<u32> = (0..logits.len() as u32).collect();
        if top_k != 0 {
            candidates.sort_by(|&lhs, &rhs| {
                let left = logits[lhs as usize];
                let right = logits[rhs as usize];
                right
                    .partial_cmp(&left)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(lhs.cmp(&rhs))
            });
            candidates.truncate(top_k);
        }

        let candidate_logits: Vec<f32> = candidates
            .iter()
            .map(|&id| logits[id as usize])
            .collect();
        let probabilities = Self::softmax(&candidate_logits, temperature)?;
        let draw = f64::from(rng.next_f32());
        let mut cumulative = 0.0f64;
        for (index, &probability) in probabilities.iter().enumerate() {
            cumulative += f64::from(probability);
            if draw < cumulative {
                return Ok(candidates[index]);
            }
        }
        Ok(*candidates.last().unwrap())
    }

    pub fn apply_repetition_penalty(
        context: &[u32],
        logits: &mut [f32],
        penalty: f32,
        window: usize,
    ) -> Result<()> {
        if !penalty.is_finite() || penalty < 1.0 {
            return Err(Error::new(
                "Repetition penalty must be finite and at least 1.0",
            ));
        }
        if penalty == 1.0 || context.is_empty() {
            return Ok(());
        }
        let first = if window != 0 && context.len() > window {
            context.len() - window
        } else {
            0
        };
        let mut penalized = vec![false; logits.len()];
        for &id in &context[first..] {
            let id = id as usize;
            if id >= logits.len() {
                return Err(Error::new(
                    "Context token ID is outside the vocabulary",
                ));
            }
            if penalized[id] {
                continue;
            }
            if !logits[id].is_finite() {
                return Err(Error::new("Logits must be finite"));
            }
            if logits[id] > 0.0 {
                logits[id] /= penalty;
            } else if logits[id] < 0.0 {
                logits[id] *= penalty;
            }
            penalized[id] = true;
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate_parameters()?;
        let mut writer = BytesWriter::new();
        writer.bytes(FILE_MAGIC);
        writer.u32(FILE_VERSION);
        writer.u64(self.vocab_size() as u64);
        writer.u64(self.embedding_dim() as u64);
        writer.u64(self.num_heads() as u64);
        writer.u64(self.num_layers() as u64);
        writer.u64(self.mlp_hidden_dim() as u64);
        writer.u64(self.max_seq_len() as u64);
        writer.f32(self.epsilon());
        writer.tensor(self.embedding.token_embeddings());
        writer.tensor(self.embedding.positional_embeddings());
        for block in &self.blocks {
            writer.tensor(block.norm1().gamma());
            writer.tensor(block.norm1().beta());
            writer.tensor(block.norm2().gamma());
            writer.tensor(block.norm2().beta());
            writer.tensor(block.attention().q_weights());
            writer.tensor(block.attention().k_weights());
            writer.tensor(block.attention().v_weights());
            writer.tensor(block.attention().output_weights());
            writer.tensor(block.fc1_weights());
            writer.tensor(block.fc2_weights());
        }
        writer.tensor(self.final_norm.gamma());
        writer.tensor(self.final_norm.beta());
        writer.tensor(&self.output_weights);
        writer.finish(path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = read_file(path)?;
        reader.magic(FILE_MAGIC)?;
        if reader.u32()? != FILE_VERSION {
            return Err(Error::new("Unsupported GPT file version"));
        }
        let vocab_size = reader.dimension()?;
        let embedding_dim = reader.dimension()?;
        let num_heads = reader.dimension()?;
        let num_layers = reader.dimension()?;
        let mlp_hidden_dim = reader.dimension()?;
        let max_seq_len = reader.dimension()?;
        if embedding_dim % num_heads != 0 {
            return Err(Error::new("Invalid GPT head configuration"));
        }
        let epsilon = reader.f32()?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::new("Invalid epsilon in GPT file"));
        }

        let token_values = vocab_size
            .checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("GPT dimensions exceed usize range"))?;
        let position_values = max_seq_len
            .checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("GPT dimensions exceed usize range"))?;
        let matrix_values = embedding_dim
            .checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("GPT dimensions exceed usize range"))?;
        let mlp_values = embedding_dim
            .checked_mul(mlp_hidden_dim)
            .ok_or_else(|| Error::new("GPT dimensions exceed usize range"))?;
        let block_values = 4usize
            .checked_mul(embedding_dim)
            .and_then(|value| value.checked_add(4 * matrix_values))
            .and_then(|value| value.checked_add(2 * mlp_values))
            .ok_or_else(|| Error::new("GPT parameter count exceeds usize range"))?;
        let total_values = token_values
            .checked_add(position_values)
            .and_then(|value| value.checked_add(num_layers * block_values))
            .and_then(|value| value.checked_add(2 * embedding_dim))
            .and_then(|value| value.checked_add(embedding_dim * vocab_size))
            .ok_or_else(|| Error::new("GPT parameter count exceeds usize range"))?;
        if reader.remaining() != total_values.saturating_mul(4) {
            return Err(Error::new("GPT dimensions do not match file size"));
        }

        let mut model = Self::new(
            vocab_size,
            embedding_dim,
            num_heads,
            num_layers,
            mlp_hidden_dim,
            max_seq_len,
            0,
            epsilon,
        )?;
        model.embedding.token_embeddings_mut().clone_from(&Tensor::new(
            [vocab_size, embedding_dim],
            reader.values(token_values)?,
        )?);
        model
            .embedding
            .positional_embeddings_mut()
            .clone_from(&Tensor::new(
                [max_seq_len, embedding_dim],
                reader.values(position_values)?,
            )?);
        for block in &mut model.blocks {
            let gamma1 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
            let beta1 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
            let gamma2 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
            let beta2 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
            block.norm1_mut().set_parameters(gamma1, beta1)?;
            block.norm2_mut().set_parameters(gamma2, beta2)?;
            let attention = crate::attention::CausalSelfAttention::from_tensors(
                num_heads,
                Tensor::new(
                    [embedding_dim, embedding_dim],
                    reader.values(matrix_values)?,
                )?,
                Tensor::new(
                    [embedding_dim, embedding_dim],
                    reader.values(matrix_values)?,
                )?,
                Tensor::new(
                    [embedding_dim, embedding_dim],
                    reader.values(matrix_values)?,
                )?,
                Tensor::new(
                    [embedding_dim, embedding_dim],
                    reader.values(matrix_values)?,
                )?,
            )?;
            block.set_attention(attention);
            block.fc1_weights_mut().clone_from(&Tensor::new(
                [embedding_dim, mlp_hidden_dim],
                reader.values(mlp_values)?,
            )?);
            block.fc2_weights_mut().clone_from(&Tensor::new(
                [mlp_hidden_dim, embedding_dim],
                reader.values(mlp_values)?,
            )?);
        }
        let final_gamma = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
        let final_beta = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
        model.final_norm.set_parameters(final_gamma, final_beta)?;
        model.output_weights = Tensor::new(
            [embedding_dim, vocab_size],
            reader.values(embedding_dim * vocab_size)?,
        )?;
        model.zero_grad();
        model.validate_parameters()?;
        Ok(model)
    }

    pub fn vocab_size(&self) -> usize {
        self.embedding.vocab_size()
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding.embedding_dim()
    }

    pub fn num_heads(&self) -> usize {
        self.blocks.first().map(|block| block.num_heads()).unwrap_or(0)
    }

    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    pub fn mlp_hidden_dim(&self) -> usize {
        self.blocks
            .first()
            .map(|block| block.mlp_hidden_dim())
            .unwrap_or(0)
    }

    pub fn max_seq_len(&self) -> usize {
        self.embedding.max_seq_len()
    }

    pub fn epsilon(&self) -> f32 {
        self.final_norm.epsilon()
    }

    pub fn parameter_count(&self) -> GPTParameterCount {
        let mut count = GPTParameterCount {
            embeddings: self.embedding.token_embeddings().size()
                + self.embedding.positional_embeddings().size(),
            ..Default::default()
        };
        for block in &self.blocks {
            count.transformer_blocks += block.norm1().gamma().size()
                + block.norm1().beta().size()
                + block.norm2().gamma().size()
                + block.norm2().beta().size()
                + block.attention().q_weights().size()
                + block.attention().k_weights().size()
                + block.attention().v_weights().size()
                + block.attention().output_weights().size()
                + block.fc1_weights().size()
                + block.fc2_weights().size();
        }
        count.final_layer_norm = self.final_norm.gamma().size() + self.final_norm.beta().size();
        count.output_projection = self.output_weights.size();
        count
    }

    pub fn zero_grad(&mut self) {
        let gradients = GPTGradients::zeros(
            &self.embedding,
            &self.blocks,
            &self.final_norm,
            &self.output_weights,
        );
        self.gradients = gradients;
    }

    pub fn gradients(&self) -> &GPTGradients {
        &self.gradients
    }

    pub fn gradients_mut(&mut self) -> &mut GPTGradients {
        &mut self.gradients
    }

    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    pub fn embedding_mut(&mut self) -> &mut Embedding {
        &mut self.embedding
    }

    pub fn blocks(&self) -> &[TransformerBlock] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut [TransformerBlock] {
        &mut self.blocks
    }

    pub fn final_norm(&self) -> &LayerNorm {
        &self.final_norm
    }

    pub fn final_norm_mut(&mut self) -> &mut LayerNorm {
        &mut self.final_norm
    }

    pub fn output_weights(&self) -> &Tensor {
        &self.output_weights
    }

    pub fn output_weights_mut(&mut self) -> &mut Tensor {
        &mut self.output_weights
    }

    pub fn parameter_refs(&self) -> Vec<ParamRef<'_>> {
        let mut parameters = Vec::new();
        parameters.push(ParamRef {
            parameter: self.embedding.token_embeddings(),
            gradient: &self.gradients.embedding.token_embeddings,
            weight_decay: true,
            name: "embedding.token".to_string(),
        });
        parameters.push(ParamRef {
            parameter: self.embedding.positional_embeddings(),
            gradient: &self.gradients.embedding.positional_embeddings,
            weight_decay: true,
            name: "embedding.position".to_string(),
        });
        for (index, block) in self.blocks.iter().enumerate() {
            let gradient = &self.gradients.blocks[index];
            let prefix = format!("block{index}.");
            parameters.push(ParamRef {
                parameter: block.norm1().gamma(),
                gradient: &gradient.norm1.gamma,
                weight_decay: false,
                name: format!("{prefix}norm1.gamma"),
            });
            parameters.push(ParamRef {
                parameter: block.norm1().beta(),
                gradient: &gradient.norm1.beta,
                weight_decay: false,
                name: format!("{prefix}norm1.beta"),
            });
            parameters.push(ParamRef {
                parameter: block.norm2().gamma(),
                gradient: &gradient.norm2.gamma,
                weight_decay: false,
                name: format!("{prefix}norm2.gamma"),
            });
            parameters.push(ParamRef {
                parameter: block.norm2().beta(),
                gradient: &gradient.norm2.beta,
                weight_decay: false,
                name: format!("{prefix}norm2.beta"),
            });
            parameters.push(ParamRef {
                parameter: block.attention().q_weights(),
                gradient: &gradient.attention.q_weights,
                weight_decay: true,
                name: format!("{prefix}q"),
            });
            parameters.push(ParamRef {
                parameter: block.attention().k_weights(),
                gradient: &gradient.attention.k_weights,
                weight_decay: true,
                name: format!("{prefix}k"),
            });
            parameters.push(ParamRef {
                parameter: block.attention().v_weights(),
                gradient: &gradient.attention.v_weights,
                weight_decay: true,
                name: format!("{prefix}v"),
            });
            parameters.push(ParamRef {
                parameter: block.attention().output_weights(),
                gradient: &gradient.attention.output_weights,
                weight_decay: true,
                name: format!("{prefix}attention.output"),
            });
            parameters.push(ParamRef {
                parameter: block.fc1_weights(),
                gradient: &gradient.fc1_weights,
                weight_decay: true,
                name: format!("{prefix}fc1"),
            });
            parameters.push(ParamRef {
                parameter: block.fc2_weights(),
                gradient: &gradient.fc2_weights,
                weight_decay: true,
                name: format!("{prefix}fc2"),
            });
        }
        parameters.push(ParamRef {
            parameter: self.final_norm.gamma(),
            gradient: &self.gradients.final_norm.gamma,
            weight_decay: false,
            name: "final_norm.gamma".to_string(),
        });
        parameters.push(ParamRef {
            parameter: self.final_norm.beta(),
            gradient: &self.gradients.final_norm.beta,
            weight_decay: false,
            name: "final_norm.beta".to_string(),
        });
        parameters.push(ParamRef {
            parameter: &self.output_weights,
            gradient: &self.gradients.output_weights,
            weight_decay: true,
            name: "output_weights".to_string(),
        });
        parameters
    }

    pub fn parameter_bindings(&mut self) -> Vec<ParamBinding<'_>> {
        let Self {
            embedding,
            blocks,
            final_norm,
            output_weights,
            gradients,
            ..
        } = self;
        let mut parameters = Vec::new();
        parameters.push(ParamBinding {
            parameter: embedding.token_embeddings_mut(),
            gradient: &gradients.embedding.token_embeddings,
            weight_decay: true,
            name: "embedding.token".to_string(),
        });
        parameters.push(ParamBinding {
            parameter: embedding.positional_embeddings_mut(),
            gradient: &gradients.embedding.positional_embeddings,
            weight_decay: true,
            name: "embedding.position".to_string(),
        });
        for (index, block) in blocks.iter_mut().enumerate() {
            let gradient = &gradients.blocks[index];
            let prefix = format!("block{index}.");
            parameters.push(ParamBinding {
                parameter: block.norm1_mut().gamma_mut(),
                gradient: &gradient.norm1.gamma,
                weight_decay: false,
                name: format!("{prefix}norm1.gamma"),
            });
            parameters.push(ParamBinding {
                parameter: block.norm1_mut().beta_mut(),
                gradient: &gradient.norm1.beta,
                weight_decay: false,
                name: format!("{prefix}norm1.beta"),
            });
            parameters.push(ParamBinding {
                parameter: block.norm2_mut().gamma_mut(),
                gradient: &gradient.norm2.gamma,
                weight_decay: false,
                name: format!("{prefix}norm2.gamma"),
            });
            parameters.push(ParamBinding {
                parameter: block.norm2_mut().beta_mut(),
                gradient: &gradient.norm2.beta,
                weight_decay: false,
                name: format!("{prefix}norm2.beta"),
            });
            parameters.push(ParamBinding {
                parameter: block.attention_mut().q_weights_mut(),
                gradient: &gradient.attention.q_weights,
                weight_decay: true,
                name: format!("{prefix}q"),
            });
            parameters.push(ParamBinding {
                parameter: block.attention_mut().k_weights_mut(),
                gradient: &gradient.attention.k_weights,
                weight_decay: true,
                name: format!("{prefix}k"),
            });
            parameters.push(ParamBinding {
                parameter: block.attention_mut().v_weights_mut(),
                gradient: &gradient.attention.v_weights,
                weight_decay: true,
                name: format!("{prefix}v"),
            });
            parameters.push(ParamBinding {
                parameter: block.attention_mut().output_weights_mut(),
                gradient: &gradient.attention.output_weights,
                weight_decay: true,
                name: format!("{prefix}attention.output"),
            });
            parameters.push(ParamBinding {
                parameter: block.fc1_weights_mut(),
                gradient: &gradient.fc1_weights,
                weight_decay: true,
                name: format!("{prefix}fc1"),
            });
            parameters.push(ParamBinding {
                parameter: block.fc2_weights_mut(),
                gradient: &gradient.fc2_weights,
                weight_decay: true,
                name: format!("{prefix}fc2"),
            });
        }
        parameters.push(ParamBinding {
            parameter: final_norm.gamma_mut(),
            gradient: &gradients.final_norm.gamma,
            weight_decay: false,
            name: "final_norm.gamma".to_string(),
        });
        parameters.push(ParamBinding {
            parameter: final_norm.beta_mut(),
            gradient: &gradients.final_norm.beta,
            weight_decay: false,
            name: "final_norm.beta".to_string(),
        });
        parameters.push(ParamBinding {
            parameter: output_weights,
            gradient: &gradients.output_weights,
            weight_decay: true,
            name: "output_weights".to_string(),
        });
        parameters
    }

    fn validate_parameters(&self) -> Result<()> {
        let vocab_size = self.embedding.vocab_size();
        let embedding_dim = self.embedding.embedding_dim();
        if self.blocks.is_empty()
            || self.embedding.token_embeddings().shape() != [vocab_size, embedding_dim]
            || self.embedding.positional_embeddings().shape()
                != [self.embedding.max_seq_len(), embedding_dim]
            || self.final_norm.gamma().shape() != [embedding_dim]
            || self.final_norm.beta().shape() != [embedding_dim]
            || self.output_weights.shape() != [embedding_dim, vocab_size]
        {
            return Err(Error::new("GPT parameters have incompatible shapes"));
        }
        let hidden_dim = self.blocks[0].mlp_hidden_dim();
        let num_heads = self.blocks[0].num_heads();
        for block in &self.blocks {
            if block.embedding_dim() != embedding_dim
                || block.num_heads() != num_heads
                || block.mlp_hidden_dim() != hidden_dim
                || block.fc1_weights().shape() != [embedding_dim, hidden_dim]
                || block.fc2_weights().shape() != [hidden_dim, embedding_dim]
            {
                return Err(Error::new(
                    "GPT Transformer Block parameters are incompatible",
                ));
            }
        }
        if self.gradients.blocks.len() != self.blocks.len() {
            return Err(Error::new("GPT gradients are not initialized"));
        }
        Ok(())
    }
}

fn component_seed(seed: u32, component: usize) -> u32 {
    seed.wrapping_add(6u32.wrapping_mul(component as u32 + 1))
}
