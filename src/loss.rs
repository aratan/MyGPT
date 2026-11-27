//! Entropía cruzada con desplazamiento de tokens y máscara.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShiftedTokens {
    pub inputs: Vec<u32>,
    pub targets: Vec<u32>,
}

pub fn shift_tokens(tokens: &[u32]) -> Result<ShiftedTokens> {
    if tokens.len() < 2 {
        return Err(Error::new(
            "At least two tokens are required for a shifted pair",
        ));
    }
    Ok(ShiftedTokens {
        inputs: tokens[..tokens.len() - 1].to_vec(),
        targets: tokens[1..].to_vec(),
    })
}

pub fn cross_entropy(logits: &Tensor, targets: &[u32], mask: &[u8]) -> Result<f32> {
    if logits.rank() != 3 {
        return Err(Error::new(
            "Logits must have shape [batch_size, seq_len, vocab_size]",
        ));
    }
    let batch_size = logits.shape()[0];
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];
    let positions = batch_size * seq_len;
    if vocab_size == 0 {
        return Err(Error::new("Vocabulary size must be greater than zero"));
    }
    if targets.len() != positions {
        return Err(Error::new("Target count must equal batch_size * seq_len"));
    }
    if !mask.is_empty() && mask.len() != positions {
        return Err(Error::new("Mask size must equal batch_size * seq_len"));
    }
    if mask.iter().any(|&value| value > 1) {
        return Err(Error::new("Mask values must be either zero or one"));
    }

    let logit_data = logits.data();
    let mut total_loss = 0.0f64;
    let mut valid_positions = 0usize;
    for position in 0..positions {
        if !mask.is_empty() && mask[position] == 0 {
            continue;
        }
        if targets[position] as usize >= vocab_size {
            return Err(Error::new("Target token ID is outside the vocabulary"));
        }

        let offset = position * vocab_size;
        let row = &logit_data[offset..offset + vocab_size];
        let mut maximum = f64::NEG_INFINITY;
        for &value in row {
            if !value.is_finite() {
                return Err(Error::new("Logits must be finite at valid positions"));
            }
            maximum = maximum.max(f64::from(value));
        }
        let mut exponential_sum = 0.0f64;
        for &value in row {
            exponential_sum += (f64::from(value) - maximum).exp();
        }
        let log_sum_exp = maximum + exponential_sum.ln();
        total_loss += log_sum_exp - f64::from(row[targets[position] as usize]);
        valid_positions += 1;
    }

    if valid_positions == 0 {
        return Err(Error::new(
            "Cross-Entropy requires at least one valid position",
        ));
    }
    Ok((total_loss / valid_positions as f64) as f32)
}

pub fn cross_entropy_backward(logits: &Tensor, targets: &[u32], mask: &[u8]) -> Result<Tensor> {
    if logits.rank() != 3 {
        return Err(Error::new(
            "Logits must have shape [batch_size, seq_len, vocab_size]",
        ));
    }
    let batch_size = logits.shape()[0];
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];
    let positions = batch_size * seq_len;
    if vocab_size == 0
        || targets.len() != positions
        || (!mask.is_empty() && mask.len() != positions)
    {
        return Err(Error::new("Incompatible shapes in Cross-Entropy backward"));
    }
    if mask.iter().any(|&value| value > 1) {
        return Err(Error::new("Mask values must be either zero or one"));
    }

    let valid_positions = (0..positions)
        .filter(|&position| mask.is_empty() || mask[position] != 0)
        .count();
    if valid_positions == 0 {
        return Err(Error::new(
            "Cross-Entropy requires at least one valid position",
        ));
    }

    let mut gradient = vec![0.0f32; logits.size()];
    let logit_data = logits.data();
    let scale = 1.0 / valid_positions as f64;
    for position in 0..positions {
        if !mask.is_empty() && mask[position] == 0 {
            continue;
        }
        if targets[position] as usize >= vocab_size {
            return Err(Error::new("Target token ID is outside the vocabulary"));
        }
        let offset = position * vocab_size;
        let row = &logit_data[offset..offset + vocab_size];
        let mut maximum = f64::NEG_INFINITY;
        for &value in row {
            if !value.is_finite() {
                return Err(Error::new("Logits must be finite at valid positions"));
            }
            maximum = maximum.max(f64::from(value));
        }
        let mut denominator = 0.0f64;
        for &value in row {
            denominator += (f64::from(value) - maximum).exp();
        }
        let target = targets[position] as usize;
        for token in 0..vocab_size {
            let probability = (f64::from(row[token]) - maximum).exp() / denominator;
            let target_value = if token == target { 1.0 } else { 0.0 };
            gradient[offset + token] = ((probability - target_value) * scale) as f32;
        }
    }
    Tensor::new(logits.shape().to_vec(), gradient)
}

pub fn perplexity(loss: f32) -> Result<f32> {
    if !loss.is_finite() || loss < 0.0 {
        return Err(Error::new(
            "Perplexity requires a finite, non-negative loss",
        ));
    }
    Ok(loss.exp())
}
