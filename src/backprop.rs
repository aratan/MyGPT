//! Retropropagación de todas las capas.

use crate::attention::{flatten_heads, CausalSelfAttention};
use crate::embedding::Embedding;
use crate::error::{Error, Result};
use crate::layer_norm::LayerNorm;
use crate::ops::gelu_backward;
use crate::tensor::Tensor;
use crate::transformer::TransformerBlock;

#[derive(Clone, Debug)]
pub struct LinearBackwardResult {
    pub input_gradient: Tensor,
    pub weight_gradient: Tensor,
}

#[derive(Clone, Debug)]
pub struct LayerNormGradients {
    pub gamma: Tensor,
    pub beta: Tensor,
}

#[derive(Clone, Debug)]
pub struct LayerNormBackwardResult {
    pub input_gradient: Tensor,
    pub parameters: LayerNormGradients,
}

#[derive(Clone, Debug)]
pub struct EmbeddingGradients {
    pub token_embeddings: Tensor,
    pub positional_embeddings: Tensor,
}

#[derive(Clone, Debug)]
pub struct AttentionGradients {
    pub q_weights: Tensor,
    pub k_weights: Tensor,
    pub v_weights: Tensor,
    pub output_weights: Tensor,
}

#[derive(Clone, Debug)]
pub struct AttentionBackwardResult {
    pub input_gradient: Tensor,
    pub parameters: AttentionGradients,
}

#[derive(Clone, Debug)]
pub struct TransformerBlockGradients {
    pub norm1: LayerNormGradients,
    pub norm2: LayerNormGradients,
    pub attention: AttentionGradients,
    pub fc1_weights: Tensor,
    pub fc2_weights: Tensor,
}

#[derive(Clone, Debug)]
pub struct TransformerBlockBackwardResult {
    pub input_gradient: Tensor,
    pub parameters: TransformerBlockGradients,
}

#[derive(Clone, Debug)]
pub struct GPTGradients {
    pub embedding: EmbeddingGradients,
    pub blocks: Vec<TransformerBlockGradients>,
    pub final_norm: LayerNormGradients,
    pub output_weights: Tensor,
}

impl GPTGradients {
    pub fn empty() -> Self {
        Self {
            embedding: EmbeddingGradients {
                token_embeddings: Tensor::zeros([0]),
                positional_embeddings: Tensor::zeros([0]),
            },
            blocks: Vec::new(),
            final_norm: LayerNormGradients {
                gamma: Tensor::zeros([0]),
                beta: Tensor::zeros([0]),
            },
            output_weights: Tensor::zeros([0]),
        }
    }

    pub fn zeros(
        embedding: &Embedding,
        blocks: &[TransformerBlock],
        final_norm: &LayerNorm,
        output_weights: &Tensor,
    ) -> Self {
        Self {
            embedding: EmbeddingGradients {
                token_embeddings: Tensor::zeros(embedding.token_embeddings().shape().to_vec()),
                positional_embeddings: Tensor::zeros(
                    embedding.positional_embeddings().shape().to_vec(),
                ),
            },
            blocks: blocks
                .iter()
                .map(|block| TransformerBlockGradients {
                    norm1: LayerNormGradients {
                        gamma: Tensor::zeros(block.norm1().gamma().shape().to_vec()),
                        beta: Tensor::zeros(block.norm1().beta().shape().to_vec()),
                    },
                    norm2: LayerNormGradients {
                        gamma: Tensor::zeros(block.norm2().gamma().shape().to_vec()),
                        beta: Tensor::zeros(block.norm2().beta().shape().to_vec()),
                    },
                    attention: AttentionGradients {
                        q_weights: Tensor::zeros(block.attention().q_weights().shape().to_vec()),
                        k_weights: Tensor::zeros(block.attention().k_weights().shape().to_vec()),
                        v_weights: Tensor::zeros(block.attention().v_weights().shape().to_vec()),
                        output_weights: Tensor::zeros(
                            block.attention().output_weights().shape().to_vec(),
                        ),
                    },
                    fc1_weights: Tensor::zeros(block.fc1_weights().shape().to_vec()),
                    fc2_weights: Tensor::zeros(block.fc2_weights().shape().to_vec()),
                })
                .collect(),
            final_norm: LayerNormGradients {
                gamma: Tensor::zeros(final_norm.gamma().shape().to_vec()),
                beta: Tensor::zeros(final_norm.beta().shape().to_vec()),
            },
            output_weights: Tensor::zeros(output_weights.shape().to_vec()),
        }
    }

    pub fn tensors(&self) -> Vec<&Tensor> {
        let mut tensors = vec![
            &self.embedding.token_embeddings,
            &self.embedding.positional_embeddings,
        ];
        for block in &self.blocks {
            tensors.push(&block.norm1.gamma);
            tensors.push(&block.norm1.beta);
            tensors.push(&block.norm2.gamma);
            tensors.push(&block.norm2.beta);
            tensors.push(&block.attention.q_weights);
            tensors.push(&block.attention.k_weights);
            tensors.push(&block.attention.v_weights);
            tensors.push(&block.attention.output_weights);
            tensors.push(&block.fc1_weights);
            tensors.push(&block.fc2_weights);
        }
        tensors.push(&self.final_norm.gamma);
        tensors.push(&self.final_norm.beta);
        tensors.push(&self.output_weights);
        tensors
    }

    pub fn tensors_mut(&mut self) -> Vec<&mut Tensor> {
        let mut tensors = vec![
            &mut self.embedding.token_embeddings,
            &mut self.embedding.positional_embeddings,
        ];
        for block in &mut self.blocks {
            tensors.push(&mut block.norm1.gamma);
            tensors.push(&mut block.norm1.beta);
            tensors.push(&mut block.norm2.gamma);
            tensors.push(&mut block.norm2.beta);
            tensors.push(&mut block.attention.q_weights);
            tensors.push(&mut block.attention.k_weights);
            tensors.push(&mut block.attention.v_weights);
            tensors.push(&mut block.attention.output_weights);
            tensors.push(&mut block.fc1_weights);
            tensors.push(&mut block.fc2_weights);
        }
        tensors.push(&mut self.final_norm.gamma);
        tensors.push(&mut self.final_norm.beta);
        tensors.push(&mut self.output_weights);
        tensors
    }
}

pub fn linear_backward(
    input: &Tensor,
    weights: &Tensor,
    output_gradient: &Tensor,
) -> Result<LinearBackwardResult> {
    if input.rank() != 3
        || weights.rank() != 2
        || input.shape()[2] != weights.shape()[0]
        || output_gradient.shape() != [input.shape()[0], input.shape()[1], weights.shape()[1]]
    {
        return Err(Error::new("Incompatible shapes in linear backward"));
    }

    let batch_size = input.shape()[0];
    let seq_len = input.shape()[1];
    let input_dim = weights.shape()[0];
    let output_dim = weights.shape()[1];
    let rows = batch_size * seq_len;

    let mut input_gradient = vec![0.0f32; rows * input_dim];
    let mut weight_gradient = vec![0.0f32; input_dim * output_dim];
    let input_data = input.data();
    let weight_data = weights.data();
    let gradient_data = output_gradient.data();

    if crate::cuda::enabled()
        && rows.saturating_mul(input_dim).saturating_mul(output_dim)
            >= crate::cuda::GPU_WORK_THRESHOLD
    {
        let weight_transposed = transpose_2d(weight_data, input_dim, output_dim);
        let input_transposed = transpose_2d(input_data, rows, input_dim);
        let input_gradient = crate::cuda::matmul(
            gradient_data,
            &weight_transposed,
            rows,
            output_dim,
            input_dim,
        )?;
        let weight_gradient = crate::cuda::matmul(
            &input_transposed,
            gradient_data,
            input_dim,
            rows,
            output_dim,
        )?;
        return Ok(LinearBackwardResult {
            input_gradient: Tensor::new(input.shape().to_vec(), input_gradient)?,
            weight_gradient: Tensor::new(weights.shape().to_vec(), weight_gradient)?,
        });
    }

    for row in 0..rows {
        let input_row = &input_data[row * input_dim..row * input_dim + input_dim];
        let gradient_row = &gradient_data[row * output_dim..row * output_dim + output_dim];
        for inner in 0..input_dim {
            let value = input_row[inner];
            for output in 0..output_dim {
                let gradient = gradient_row[output];
                weight_gradient[inner * output_dim + output] += value * gradient;
                input_gradient[row * input_dim + inner] +=
                    gradient * weight_data[inner * output_dim + output];
            }
        }
    }

    Ok(LinearBackwardResult {
        input_gradient: Tensor::new(input.shape().to_vec(), input_gradient)?,
        weight_gradient: Tensor::new(weights.shape().to_vec(), weight_gradient)?,
    })
}

pub fn layer_norm_backward(
    layer_norm: &LayerNorm,
    input: &Tensor,
    output_gradient: &Tensor,
) -> Result<LayerNormBackwardResult> {
    let dimension = layer_norm.normalized_dim();
    if input.rank() == 0
        || input.shape().last() != Some(&dimension)
        || output_gradient.shape() != input.shape()
        || layer_norm.gamma().shape() != [dimension]
        || layer_norm.beta().shape() != [dimension]
    {
        return Err(Error::new("Incompatible shapes in LayerNorm backward"));
    }

    let rows = input.size() / dimension;
    let input_data = input.data();
    let gradient_data = output_gradient.data();
    let gamma_data = layer_norm.gamma().data();
    let epsilon = f64::from(layer_norm.epsilon());
    let mut input_gradient = vec![0.0f32; input.size()];
    let mut gamma_gradient = vec![0.0f32; dimension];
    let mut beta_gradient = vec![0.0f32; dimension];
    let inverse_dimension = 1.0f64 / dimension as f64;

    let mut normalized = vec![0.0f64; dimension];
    let mut normalized_gradient = vec![0.0f64; dimension];

    for row in 0..rows {
        let offset = row * dimension;
        let values = &input_data[offset..offset + dimension];
        let mut sum = 0.0f64;
        for &value in values {
            sum += f64::from(value);
        }
        let mean = sum * inverse_dimension;

        let mut squared_differences = 0.0f64;
        for &value in values {
            let difference = f64::from(value) - mean;
            squared_differences += difference * difference;
        }
        let variance = squared_differences * inverse_dimension;
        let inverse_standard_deviation = 1.0 / (variance + epsilon).sqrt();

        let mut gradient_sum = 0.0f64;
        let mut gradient_normalized_sum = 0.0f64;
        for index in 0..dimension {
            normalized[index] = (f64::from(values[index]) - mean) * inverse_standard_deviation;
            let gradient = f64::from(gradient_data[offset + index]);
            gamma_gradient[index] += (gradient * normalized[index]) as f32;
            beta_gradient[index] += gradient as f32;
            normalized_gradient[index] = gradient * f64::from(gamma_data[index]);
            gradient_sum += normalized_gradient[index];
            gradient_normalized_sum += normalized_gradient[index] * normalized[index];
        }

        for index in 0..dimension {
            let value = inverse_standard_deviation
                * inverse_dimension
                * (dimension as f64 * normalized_gradient[index]
                    - gradient_sum
                    - normalized[index] * gradient_normalized_sum);
            input_gradient[offset + index] = value as f32;
        }
    }

    Ok(LayerNormBackwardResult {
        input_gradient: Tensor::new(input.shape().to_vec(), input_gradient)?,
        parameters: LayerNormGradients {
            gamma: Tensor::new([dimension], gamma_gradient)?,
            beta: Tensor::new([dimension], beta_gradient)?,
        },
    })
}

pub fn embedding_backward(
    embedding: &Embedding,
    token_ids: &[u32],
    batch_size: usize,
    seq_len: usize,
    output_gradient: &Tensor,
) -> Result<EmbeddingGradients> {
    let dimension = embedding.embedding_dim();
    if seq_len > embedding.max_seq_len()
        || token_ids.len() != batch_size * seq_len
        || output_gradient.shape() != [batch_size, seq_len, dimension]
    {
        return Err(Error::new("Incompatible shapes in embedding backward"));
    }

    let mut token_gradient = vec![0.0f32; embedding.token_embeddings().size()];
    let mut position_gradient = vec![0.0f32; embedding.positional_embeddings().size()];
    let gradient_data = output_gradient.data();

    for batch in 0..batch_size {
        for position in 0..seq_len {
            let index = batch * seq_len + position;
            let id = token_ids[index] as usize;
            if id >= embedding.vocab_size() {
                return Err(Error::new("Token ID is outside the embedding vocabulary"));
            }
            let source = index * dimension;
            for i in 0..dimension {
                let gradient = gradient_data[source + i];
                token_gradient[id * dimension + i] += gradient;
                position_gradient[position * dimension + i] += gradient;
            }
        }
    }

    Ok(EmbeddingGradients {
        token_embeddings: Tensor::new(
            embedding.token_embeddings().shape().to_vec(),
            token_gradient,
        )?,
        positional_embeddings: Tensor::new(
            embedding.positional_embeddings().shape().to_vec(),
            position_gradient,
        )?,
    })
}

pub fn attention_backward(
    attention: &CausalSelfAttention,
    input: &Tensor,
    output_gradient: &Tensor,
) -> Result<AttentionBackwardResult> {
    if input.rank() != 3
        || input.shape()[2] != attention.embedding_dim()
        || output_gradient.shape() != input.shape()
    {
        return Err(Error::new("Incompatible shapes in attention backward"));
    }

    let forward = attention.forward_with_details(input, false)?;
    let output_backward = linear_backward(
        &forward.context,
        attention.output_weights(),
        output_gradient,
    )?;
    let batch_size = input.shape()[0];
    let seq_len = input.shape()[1];
    let embedding_dim = attention.embedding_dim();
    let num_heads = attention.num_heads();
    let head_dim = attention.head_dim();

    let mut queries_gradient = vec![0.0f32; forward.queries.size()];
    let mut keys_gradient = vec![0.0f32; forward.keys.size()];
    let mut values_gradient = vec![0.0f32; forward.values.size()];
    let context_gradient = output_backward.input_gradient.data();
    let probabilities = forward.probabilities.data();
    let value_data = forward.values.data();
    let query_data = forward.queries.data();
    let key_data = forward.keys.data();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut probability_gradients = vec![0.0f64; seq_len];

    for batch in 0..batch_size {
        for head in 0..num_heads {
            let head_base = (batch * num_heads + head) * seq_len;
            for query in 0..seq_len {
                probability_gradients.fill(0.0);
                let query_offset = (head_base + query) * head_dim;
                let probability_offset = (head_base + query) * seq_len;
                let context_offset = (batch * seq_len + query) * embedding_dim + head * head_dim;

                for key in 0..=query {
                    let value_offset = (head_base + key) * head_dim;
                    let probability = probabilities[probability_offset + key];
                    for dimension in 0..head_dim {
                        let context_gradient_value = context_gradient[context_offset + dimension];
                        let value = value_data[value_offset + dimension];
                        values_gradient[value_offset + dimension] +=
                            probability * context_gradient_value;
                        probability_gradients[key] +=
                            f64::from(context_gradient_value) * f64::from(value);
                    }
                }

                let mut weighted_probability_gradient = 0.0f64;
                for key in 0..=query {
                    weighted_probability_gradient += probability_gradients[key]
                        * f64::from(probabilities[probability_offset + key]);
                }
                for key in 0..=query {
                    let probability = probabilities[probability_offset + key];
                    let score_gradient = f64::from(probability)
                        * (probability_gradients[key] - weighted_probability_gradient)
                        * f64::from(scale);
                    let key_offset = (head_base + key) * head_dim;
                    for dimension in 0..head_dim {
                        queries_gradient[query_offset + dimension] +=
                            (score_gradient * f64::from(key_data[key_offset + dimension])) as f32;
                        keys_gradient[key_offset + dimension] += (score_gradient
                            * f64::from(query_data[query_offset + dimension]))
                            as f32;
                    }
                }
            }
        }
    }

    let flat_queries = flatten_heads(
        &Tensor::new(forward.queries.shape().to_vec(), queries_gradient)?,
        num_heads,
        head_dim,
    )?;
    let flat_keys = flatten_heads(
        &Tensor::new(forward.keys.shape().to_vec(), keys_gradient)?,
        num_heads,
        head_dim,
    )?;
    let flat_values = flatten_heads(
        &Tensor::new(forward.values.shape().to_vec(), values_gradient)?,
        num_heads,
        head_dim,
    )?;

    let query_backward = linear_backward(input, attention.q_weights(), &flat_queries)?;
    let key_backward = linear_backward(input, attention.k_weights(), &flat_keys)?;
    let value_backward = linear_backward(input, attention.v_weights(), &flat_values)?;
    let input_gradient = query_backward
        .input_gradient
        .add(&key_backward.input_gradient)?
        .add(&value_backward.input_gradient)?;

    Ok(AttentionBackwardResult {
        input_gradient,
        parameters: AttentionGradients {
            q_weights: query_backward.weight_gradient,
            k_weights: key_backward.weight_gradient,
            v_weights: value_backward.weight_gradient,
            output_weights: output_backward.weight_gradient,
        },
    })
}

pub fn transformer_block_backward(
    block: &TransformerBlock,
    input: &Tensor,
    output_gradient: &Tensor,
) -> Result<TransformerBlockBackwardResult> {
    if input.rank() != 3
        || input.shape()[2] != block.embedding_dim()
        || output_gradient.shape() != input.shape()
    {
        return Err(Error::new(
            "Incompatible shapes in Transformer Block backward",
        ));
    }

    let normalized_attention = block.norm1().forward(input)?;
    let residual = input.add(&block.attention().forward(&normalized_attention)?)?;
    let normalized_mlp = block.norm2().forward(&residual)?;
    let hidden = crate::ops::linear(&normalized_mlp, block.fc1_weights())?;
    let activated = crate::ops::gelu(&hidden);

    let fc2_backward = linear_backward(&activated, block.fc2_weights(), output_gradient)?;
    let hidden_gradient = gelu_backward(&hidden, &fc2_backward.input_gradient)?;
    let fc1_backward = linear_backward(&normalized_mlp, block.fc1_weights(), &hidden_gradient)?;
    let norm2_backward =
        layer_norm_backward(block.norm2(), &residual, &fc1_backward.input_gradient)?;
    let residual_gradient = output_gradient.add(&norm2_backward.input_gradient)?;

    let attention_backward_result =
        attention_backward(block.attention(), &normalized_attention, &residual_gradient)?;
    let norm1_backward = layer_norm_backward(
        block.norm1(),
        input,
        &attention_backward_result.input_gradient,
    )?;
    let input_gradient = residual_gradient.add(&norm1_backward.input_gradient)?;

    Ok(TransformerBlockBackwardResult {
        input_gradient,
        parameters: TransformerBlockGradients {
            norm1: norm1_backward.parameters,
            norm2: norm2_backward.parameters,
            attention: attention_backward_result.parameters,
            fc1_weights: fc1_backward.weight_gradient,
            fc2_weights: fc2_backward.weight_gradient,
        },
    })
}

pub fn accumulate(target: &mut Tensor, source: &Tensor) -> Result<()> {
    target.add_assign(source)
}

pub fn accumulate_layer_norm(
    target: &mut LayerNormGradients,
    source: &LayerNormGradients,
) -> Result<()> {
    target.gamma.add_assign(&source.gamma)?;
    target.beta.add_assign(&source.beta)?;
    Ok(())
}

pub fn accumulate_attention(
    target: &mut AttentionGradients,
    source: &AttentionGradients,
) -> Result<()> {
    target.q_weights.add_assign(&source.q_weights)?;
    target.k_weights.add_assign(&source.k_weights)?;
    target.v_weights.add_assign(&source.v_weights)?;
    target.output_weights.add_assign(&source.output_weights)?;
    Ok(())
}

pub fn accumulate_block(
    target: &mut TransformerBlockGradients,
    source: &TransformerBlockGradients,
) -> Result<()> {
    accumulate_layer_norm(&mut target.norm1, &source.norm1)?;
    accumulate_layer_norm(&mut target.norm2, &source.norm2)?;
    accumulate_attention(&mut target.attention, &source.attention)?;
    target.fc1_weights.add_assign(&source.fc1_weights)?;
    target.fc2_weights.add_assign(&source.fc2_weights)?;
    Ok(())
}

fn transpose_2d(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            output[column * rows + row] = values[row * columns + column];
        }
    }
    output
}
