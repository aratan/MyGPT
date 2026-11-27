//! Bloque transformer pre-norm con atención causal y MLP GELU.

use crate::attention::CausalSelfAttention;
use crate::error::{Error, Result};
use crate::layer_norm::LayerNorm;
use crate::ops::{gelu, linear};
use crate::serialize::{read_file, BytesWriter};
use crate::tensor::Tensor;
use std::path::Path;

const INITIALIZATION_SCALE: f32 = 0.02;
const FILE_MAGIC: &[u8; 8] = b"MINITRFB";
const FILE_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct TransformerBlock {
    mlp_hidden_dim: usize,
    attention: CausalSelfAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    fc1_weights: Tensor,
    fc2_weights: Tensor,
}

impl TransformerBlock {
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        mlp_hidden_dim: usize,
        seed: u32,
        epsilon: f32,
    ) -> Result<Self> {
        if mlp_hidden_dim == 0 {
            return Err(Error::new("mlp_hidden_dim must be greater than zero"));
        }
        Ok(Self {
            mlp_hidden_dim,
            attention: CausalSelfAttention::new(embedding_dim, num_heads, seed)?,
            norm1: LayerNorm::new(embedding_dim, epsilon)?,
            norm2: LayerNorm::new(embedding_dim, epsilon)?,
            fc1_weights: Tensor::random(
                [embedding_dim, mlp_hidden_dim],
                seed.wrapping_add(4),
                -INITIALIZATION_SCALE,
                INITIALIZATION_SCALE,
            )?,
            fc2_weights: Tensor::random(
                [mlp_hidden_dim, embedding_dim],
                seed.wrapping_add(5),
                -INITIALIZATION_SCALE,
                INITIALIZATION_SCALE,
            )?,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.validate_parameters()?;
        if input.rank() != 3 || input.shape()[2] != self.embedding_dim() {
            return Err(Error::new(
                "Transformer Block input must have shape [batch_size, seq_len, embedding_dim]",
            ));
        }

        let normalized_attention = self.norm1.forward(input)?;
        let attention_output = self.attention.forward(&normalized_attention)?;
        let residual = input.add(&attention_output)?;

        let normalized_mlp = self.norm2.forward(&residual)?;
        let hidden = linear(&normalized_mlp, &self.fc1_weights)?;
        let activated = gelu(&hidden);
        let mlp_output = linear(&activated, &self.fc2_weights)?;
        residual.add(&mlp_output)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate_parameters()?;
        let mut writer = BytesWriter::new();
        writer.bytes(FILE_MAGIC);
        writer.u32(FILE_VERSION);
        writer.u64(self.embedding_dim() as u64);
        writer.u64(self.num_heads() as u64);
        writer.u64(self.mlp_hidden_dim as u64);
        writer.f32(self.epsilon());
        writer.tensor(self.norm1.gamma());
        writer.tensor(self.norm1.beta());
        writer.tensor(self.norm2.gamma());
        writer.tensor(self.norm2.beta());
        writer.tensor(self.attention.q_weights());
        writer.tensor(self.attention.k_weights());
        writer.tensor(self.attention.v_weights());
        writer.tensor(self.attention.output_weights());
        writer.tensor(&self.fc1_weights);
        writer.tensor(&self.fc2_weights);
        writer.finish(path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = read_file(path)?;
        reader.magic(FILE_MAGIC)?;
        if reader.u32()? != FILE_VERSION {
            return Err(Error::new("Unsupported Transformer Block file version"));
        }
        let embedding_dim = reader.dimension()?;
        let num_heads = reader.dimension()?;
        let mlp_hidden_dim = reader.dimension()?;
        if embedding_dim % num_heads != 0 {
            return Err(Error::new(
                "Invalid Transformer Block head configuration",
            ));
        }
        let epsilon = reader.f32()?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::new(
                "Invalid epsilon in Transformer Block file",
            ));
        }

        let matrix_size = embedding_dim
            .checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("Transformer Block dimensions exceed usize range"))?;
        let mlp_matrix_size = embedding_dim
            .checked_mul(mlp_hidden_dim)
            .ok_or_else(|| Error::new("Transformer Block dimensions exceed usize range"))?;
        let total_values = 4usize
            .checked_mul(embedding_dim)
            .and_then(|value| value.checked_add(4 * matrix_size))
            .and_then(|value| value.checked_add(2 * mlp_matrix_size))
            .ok_or_else(|| Error::new("Transformer Block parameter count exceeds usize range"))?;
        if reader.remaining() != total_values.saturating_mul(4) {
            return Err(Error::new(
                "Transformer Block dimensions do not match file size",
            ));
        }

        let mut block = Self::new(embedding_dim, num_heads, mlp_hidden_dim, 0, epsilon)?;
        let gamma1 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
        let beta1 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
        let gamma2 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
        let beta2 = Tensor::new([embedding_dim], reader.values(embedding_dim)?)?;
        block.norm1.set_parameters(gamma1, beta1)?;
        block.norm2.set_parameters(gamma2, beta2)?;

        let mut read_matrix = |rows: usize, columns: usize, count: usize| -> Result<Tensor> {
            Tensor::new([rows, columns], reader.values(count)?)
        };
        block.attention = CausalSelfAttention::from_tensors(
            num_heads,
            read_matrix(embedding_dim, embedding_dim, matrix_size)?,
            read_matrix(embedding_dim, embedding_dim, matrix_size)?,
            read_matrix(embedding_dim, embedding_dim, matrix_size)?,
            read_matrix(embedding_dim, embedding_dim, matrix_size)?,
        )?;
        block.fc1_weights = read_matrix(embedding_dim, mlp_hidden_dim, mlp_matrix_size)?;
        block.fc2_weights = read_matrix(mlp_hidden_dim, embedding_dim, mlp_matrix_size)?;
        block.validate_parameters()?;
        Ok(block)
    }

    pub fn embedding_dim(&self) -> usize {
        self.attention.embedding_dim()
    }

    pub fn num_heads(&self) -> usize {
        self.attention.num_heads()
    }

    pub fn mlp_hidden_dim(&self) -> usize {
        self.mlp_hidden_dim
    }

    pub fn epsilon(&self) -> f32 {
        self.norm1.epsilon()
    }

    pub fn attention(&self) -> &CausalSelfAttention {
        &self.attention
    }

    pub fn attention_mut(&mut self) -> &mut CausalSelfAttention {
        &mut self.attention
    }

    pub fn set_attention(&mut self, attention: CausalSelfAttention) {
        self.attention = attention;
    }

    pub fn norm1(&self) -> &LayerNorm {
        &self.norm1
    }

    pub fn norm1_mut(&mut self) -> &mut LayerNorm {
        &mut self.norm1
    }

    pub fn norm2(&self) -> &LayerNorm {
        &self.norm2
    }

    pub fn norm2_mut(&mut self) -> &mut LayerNorm {
        &mut self.norm2
    }

    pub fn fc1_weights(&self) -> &Tensor {
        &self.fc1_weights
    }

    pub fn fc1_weights_mut(&mut self) -> &mut Tensor {
        &mut self.fc1_weights
    }

    pub fn fc2_weights(&self) -> &Tensor {
        &self.fc2_weights
    }

    pub fn fc2_weights_mut(&mut self) -> &mut Tensor {
        &mut self.fc2_weights
    }

    fn validate_parameters(&self) -> Result<()> {
        let dimension = self.embedding_dim();
        if !self.attention.q_weights().shape().eq(&[dimension, dimension]) {
            return Err(Error::new(
                "Transformer Block parameters have incompatible shapes",
            ));
        }
        if self.fc1_weights.shape() != [dimension, self.mlp_hidden_dim]
            || self.fc2_weights.shape() != [self.mlp_hidden_dim, dimension]
        {
            return Err(Error::new(
                "Transformer Block parameters have incompatible shapes",
            ));
        }
        Ok(())
    }
}
