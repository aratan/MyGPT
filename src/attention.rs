//! Self-attention causal multi-cabeza.

use crate::error::{Error, Result};
use crate::ops::linear;
use crate::serialize::{read_file, BytesWriter};
use crate::tensor::Tensor;
use std::path::Path;

const INITIALIZATION_SCALE: f32 = 0.02;
const FILE_MAGIC: &[u8; 8] = b"MINIATTN";
const FILE_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct AttentionForward {
    pub output: Tensor,
    pub context: Tensor,
    pub queries: Tensor,
    pub keys: Tensor,
    pub values: Tensor,
    pub logits: Tensor,
    pub probabilities: Tensor,
}

#[derive(Clone, Debug)]
pub struct CausalSelfAttention {
    embedding_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_weights: Tensor,
    k_weights: Tensor,
    v_weights: Tensor,
    output_weights: Tensor,
}

impl CausalSelfAttention {
    pub fn new(embedding_dim: usize, num_heads: usize, seed: u32) -> Result<Self> {
        if embedding_dim == 0 {
            return Err(Error::new("embedding_dim must be greater than zero"));
        }
        if num_heads == 0 {
            return Err(Error::new("num_heads must be greater than zero"));
        }
        if embedding_dim % num_heads != 0 {
            return Err(Error::new("embedding_dim must be divisible by num_heads"));
        }
        let matrix = || [embedding_dim, embedding_dim];
        let random =
            |seed: u32| Tensor::random(matrix(), seed, -INITIALIZATION_SCALE, INITIALIZATION_SCALE);
        Ok(Self {
            embedding_dim,
            num_heads,
            head_dim: embedding_dim / num_heads,
            q_weights: random(seed)?,
            k_weights: random(seed.wrapping_add(1))?,
            v_weights: random(seed.wrapping_add(2))?,
            output_weights: random(seed.wrapping_add(3))?,
        })
    }

    pub fn from_tensors(
        num_heads: usize,
        q_weights: Tensor,
        k_weights: Tensor,
        v_weights: Tensor,
        output_weights: Tensor,
    ) -> Result<Self> {
        if q_weights.rank() != 2
            || k_weights.rank() != 2
            || v_weights.rank() != 2
            || output_weights.rank() != 2
        {
            return Err(Error::new("Attention parameters must be matrices"));
        }
        let embedding_dim = q_weights.shape()[0];
        let expected = [embedding_dim, embedding_dim];
        if embedding_dim == 0
            || num_heads == 0
            || q_weights.shape() != expected
            || k_weights.shape() != expected
            || v_weights.shape() != expected
            || output_weights.shape() != expected
            || embedding_dim % num_heads != 0
        {
            return Err(Error::new("Invalid attention parameter dimensions"));
        }
        Ok(Self {
            embedding_dim,
            num_heads,
            head_dim: embedding_dim / num_heads,
            q_weights,
            k_weights,
            v_weights,
            output_weights,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_details(input, false)?.output)
    }

    pub fn forward_with_details(
        &self,
        input: &Tensor,
        include_logits: bool,
    ) -> Result<AttentionForward> {
        self.validate_parameter_shapes()?;
        if input.rank() != 3 || input.shape()[2] != self.embedding_dim {
            return Err(Error::new(
                "Attention input must have shape [batch_size, seq_len, embedding_dim]",
            ));
        }
        let queries = split_heads(
            &linear(input, &self.q_weights)?,
            self.num_heads,
            self.head_dim,
        )?;
        let keys = split_heads(
            &linear(input, &self.k_weights)?,
            self.num_heads,
            self.head_dim,
        )?;
        let values = split_heads(
            &linear(input, &self.v_weights)?,
            self.num_heads,
            self.head_dim,
        )?;
        self.attend(queries, keys, values, include_logits)
    }

    pub fn forward_from_qkv(
        &self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
    ) -> Result<AttentionForward> {
        self.validate_parameter_shapes()?;
        if queries.rank() != 4
            || keys.rank() != 4
            || values.rank() != 4
            || queries.shape() != keys.shape()
            || queries.shape() != values.shape()
            || queries.shape()[1] != self.num_heads
            || queries.shape()[3] != self.head_dim
        {
            return Err(Error::new(
                "Q, K and V must have matching shape [batch_size, num_heads, seq_len, head_dim]",
            ));
        }
        self.attend(queries, keys, values, true)
    }

    fn attend(
        &self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        include_logits: bool,
    ) -> Result<AttentionForward> {
        let batch_size = queries.shape()[0];
        let seq_len = queries.shape()[2];
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let embedding_dim = self.embedding_dim;

        let mut logits = if include_logits {
            vec![f32::NEG_INFINITY; batch_size * num_heads * seq_len * seq_len]
        } else {
            Vec::new()
        };
        let mut probabilities = vec![0.0f32; batch_size * num_heads * seq_len * seq_len];
        let mut combined = vec![0.0f32; batch_size * seq_len * embedding_dim];

        let query_data = queries.data();
        let key_data = keys.data();
        let value_data = values.data();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; seq_len];

        for batch in 0..batch_size {
            for head in 0..num_heads {
                let head_base = (batch * num_heads + head) * seq_len;
                for query in 0..seq_len {
                    let query_offset = (head_base + query) * head_dim;
                    let probability_offset = (head_base + query) * seq_len;

                    // Causal: cada token solo atiende a sí mismo y a los anteriores.
                    for key in 0..=query {
                        let key_offset = (head_base + key) * head_dim;
                        let mut score = 0.0f32;
                        for dimension in 0..head_dim {
                            score += query_data[query_offset + dimension]
                                * key_data[key_offset + dimension];
                        }
                        scores[key] = score * scale;
                        if include_logits {
                            logits[probability_offset + key] = scores[key];
                        }
                    }

                    let mut maximum = scores[0];
                    for &score in &scores[1..=query] {
                        maximum = maximum.max(score);
                    }
                    let mut denominator = 0.0f32;
                    for key in 0..=query {
                        let weight = (scores[key] - maximum).exp();
                        probabilities[probability_offset + key] = weight;
                        denominator += weight;
                    }

                    let output_offset = (batch * seq_len + query) * embedding_dim + head * head_dim;
                    for key in 0..=query {
                        let weight = probabilities[probability_offset + key] / denominator;
                        probabilities[probability_offset + key] = weight;
                        let value_offset = (head_base + key) * head_dim;
                        for dimension in 0..head_dim {
                            combined[output_offset + dimension] +=
                                weight * value_data[value_offset + dimension];
                        }
                    }
                }
            }
        }

        let combined = Tensor::new(vec![batch_size, seq_len, embedding_dim], combined)?;
        let output = linear(&combined, &self.output_weights)?;
        let logits = if include_logits {
            Tensor::new(vec![batch_size, num_heads, seq_len, seq_len], logits)?
        } else {
            Tensor::zeros([0])
        };
        let probabilities =
            Tensor::new(vec![batch_size, num_heads, seq_len, seq_len], probabilities)?;

        Ok(AttentionForward {
            output,
            context: combined,
            queries,
            keys,
            values,
            logits,
            probabilities,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate_parameter_shapes()?;
        let mut writer = BytesWriter::new();
        writer.bytes(FILE_MAGIC);
        writer.u32(FILE_VERSION);
        writer.u64(self.embedding_dim as u64);
        writer.u64(self.num_heads as u64);
        writer.tensor(&self.q_weights);
        writer.tensor(&self.k_weights);
        writer.tensor(&self.v_weights);
        writer.tensor(&self.output_weights);
        writer.finish(path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = read_file(path)?;
        reader.magic(FILE_MAGIC)?;
        if reader.u32()? != FILE_VERSION {
            return Err(Error::new("Unsupported attention file version"));
        }
        let embedding_dim = reader.dimension()?;
        let num_heads = reader.dimension()?;
        if embedding_dim % num_heads != 0 {
            return Err(Error::new("Invalid attention head configuration"));
        }
        let matrix_size = embedding_dim
            .checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("Attention dimensions exceed usize range"))?;
        let expected = matrix_size
            .checked_mul(16)
            .ok_or_else(|| Error::new("Attention data size exceeds file format range"))?;
        if reader.remaining() != expected {
            return Err(Error::new("Attention dimensions do not match file size"));
        }
        let matrix = |reader: &mut crate::serialize::BytesReader| -> Result<Tensor> {
            Tensor::new([embedding_dim, embedding_dim], reader.values(matrix_size)?)
        };
        let q_weights = matrix(&mut reader)?;
        let k_weights = matrix(&mut reader)?;
        let v_weights = matrix(&mut reader)?;
        let output_weights = matrix(&mut reader)?;
        Self::from_tensors(num_heads, q_weights, k_weights, v_weights, output_weights)
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn q_weights(&self) -> &Tensor {
        &self.q_weights
    }

    pub fn q_weights_mut(&mut self) -> &mut Tensor {
        &mut self.q_weights
    }

    pub fn k_weights(&self) -> &Tensor {
        &self.k_weights
    }

    pub fn k_weights_mut(&mut self) -> &mut Tensor {
        &mut self.k_weights
    }

    pub fn v_weights(&self) -> &Tensor {
        &self.v_weights
    }

    pub fn v_weights_mut(&mut self) -> &mut Tensor {
        &mut self.v_weights
    }

    pub fn output_weights(&self) -> &Tensor {
        &self.output_weights
    }

    pub fn output_weights_mut(&mut self) -> &mut Tensor {
        &mut self.output_weights
    }

    fn validate_parameter_shapes(&self) -> Result<()> {
        let expected = [self.embedding_dim, self.embedding_dim];
        if self.q_weights.shape() != expected
            || self.k_weights.shape() != expected
            || self.v_weights.shape() != expected
            || self.output_weights.shape() != expected
        {
            return Err(Error::new(
                "Attention parameter shapes do not match the layer configuration",
            ));
        }
        Ok(())
    }
}

/// `[batch, seq, embedding] -> [batch, heads, seq, head_dim]`.
pub fn split_heads(input: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let batch_size = input.shape()[0];
    let seq_len = input.shape()[1];
    let embedding_dim = input.shape()[2];
    let mut output = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];
    let input_data = input.data();
    for batch in 0..batch_size {
        for head in 0..num_heads {
            for position in 0..seq_len {
                let source = (batch * seq_len + position) * embedding_dim + head * head_dim;
                let destination = ((batch * num_heads + head) * seq_len + position) * head_dim;
                output[destination..destination + head_dim]
                    .copy_from_slice(&input_data[source..source + head_dim]);
            }
        }
    }
    Tensor::new(vec![batch_size, num_heads, seq_len, head_dim], output)
}

/// `[batch, heads, seq, head_dim] -> [batch, seq, embedding]`.
pub fn flatten_heads(input: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let batch_size = input.shape()[0];
    let seq_len = input.shape()[2];
    let embedding_dim = num_heads * head_dim;
    let mut output = vec![0.0f32; batch_size * seq_len * embedding_dim];
    let input_data = input.data();
    for batch in 0..batch_size {
        for head in 0..num_heads {
            for position in 0..seq_len {
                let head_offset = ((batch * num_heads + head) * seq_len + position) * head_dim;
                let flat_offset = (batch * seq_len + position) * embedding_dim + head * head_dim;
                output[flat_offset..flat_offset + head_dim]
                    .copy_from_slice(&input_data[head_offset..head_offset + head_dim]);
            }
        }
    }
    Tensor::new(vec![batch_size, seq_len, embedding_dim], output)
}
