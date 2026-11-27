//! Análisis de probabilidades y evaluación determinista de continuaciones.

use crate::error::{Error, Result};
use crate::gpt::GPT;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenProbability {
    pub id: u32,
    pub probability: f32,
    pub rank: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContinuationEvaluation {
    pub loss: f64,
    pub tokens: Vec<TokenProbability>,
}

fn ranked_probabilities(model: &GPT, context: &[u32]) -> Result<Vec<TokenProbability>> {
    if context.is_empty() {
        return Err(Error::new(
            "Probability analysis requires a non-empty context",
        ));
    }
    let logits = model.forward(context, 1, context.len())?;
    let vocabulary = model.vocab_size();
    let start = (context.len() - 1) * vocabulary;
    let last_logits = &logits.data()[start..start + vocabulary];
    let probabilities = GPT::softmax(last_logits, 1.0)?;
    let mut ranked: Vec<TokenProbability> = probabilities
        .iter()
        .enumerate()
        .map(|(index, &probability)| TokenProbability {
            id: index as u32,
            probability,
            rank: 0,
        })
        .collect();
    ranked.sort_by(|lhs, rhs| {
        rhs.probability
            .partial_cmp(&lhs.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(lhs.id.cmp(&rhs.id))
    });
    for (index, item) in ranked.iter_mut().enumerate() {
        item.rank = index + 1;
    }
    Ok(ranked)
}

pub fn top_token_probabilities(
    model: &GPT,
    context: &[u32],
    count: usize,
) -> Result<Vec<TokenProbability>> {
    let mut ranked = ranked_probabilities(model, context)?;
    if count == 0 || count > ranked.len() {
        return Err(Error::new(
            "Top count must be between 1 and vocabulary size",
        ));
    }
    ranked.truncate(count);
    Ok(ranked)
}

pub fn evaluate_continuation(
    model: &GPT,
    prompt: &[u32],
    expected: &[u32],
) -> Result<ContinuationEvaluation> {
    if prompt.is_empty() || expected.is_empty() {
        return Err(Error::new(
            "Continuation evaluation requires prompt and expected text",
        ));
    }
    let mut sequence = Vec::with_capacity(prompt.len() + expected.len());
    sequence.extend_from_slice(prompt);
    sequence.extend_from_slice(expected);

    let mut evaluation = ContinuationEvaluation {
        loss: 0.0,
        tokens: Vec::with_capacity(expected.len()),
    };
    for (index, &target) in expected.iter().enumerate() {
        let context_size = prompt.len() + index;
        let start = if context_size > model.max_seq_len() {
            context_size - model.max_seq_len()
        } else {
            0
        };
        let context = &sequence[start..context_size];
        let ranked = ranked_probabilities(model, context)?;
        if target as usize >= model.vocab_size() {
            return Err(Error::new("Expected token ID is outside the vocabulary"));
        }
        let entry = ranked
            .iter()
            .find(|item| item.id == target)
            .ok_or_else(|| Error::new("Expected token is missing from the ranking"))?;
        evaluation.tokens.push(TokenProbability {
            id: target,
            probability: entry.probability,
            rank: entry.rank,
        });
        evaluation.loss -= f64::from(entry.probability).max(1.0e-300).ln();
    }
    evaluation.loss /= expected.len() as f64;
    Ok(evaluation)
}
