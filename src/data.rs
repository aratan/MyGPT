//! Corpus, UTF-8, separación train/validation y batches de secuencias.

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::tokenizer::BpeTokenizer;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorpusSplit {
    pub training: String,
    pub validation: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataLoaderState {
    pub training_order: Vec<usize>,
    pub validation_order: Vec<usize>,
    pub training_cursor: usize,
    pub validation_cursor: usize,
    pub generator_state: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceBatch {
    pub inputs: Vec<u32>,
    pub targets: Vec<u32>,
    pub mask: Vec<u8>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub real_tokens: usize,
}

pub fn utf8_codepoint_count(text: &str) -> usize {
    text.chars().count()
}

/// Separa el corpus por puntos de código (no parte caracteres multibyte).
pub fn split_corpus_utf8(text: &str, validation_ratio: f64) -> Result<CorpusSplit> {
    if !validation_ratio.is_finite() || validation_ratio <= 0.0 || validation_ratio >= 1.0 {
        return Err(Error::new("Validation ratio must be between zero and one"));
    }
    let count = utf8_codepoint_count(text);
    let training_count = (count as f64 * (1.0 - validation_ratio)) as usize;
    if training_count == 0 || training_count == count {
        return Err(Error::new("Both corpus splits must be non-empty"));
    }
    let byte_offset = text
        .char_indices()
        .nth(training_count)
        .map(|(offset, _)| offset)
        .unwrap_or(text.len());
    Ok(CorpusSplit {
        training: text[..byte_offset].to_string(),
        validation: text[byte_offset..].to_string(),
    })
}

pub fn read_utf8_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|error| Error::new(format!("Cannot open corpus {}: {error}", path.display())))?;
    String::from_utf8(bytes)
        .map_err(|_| Error::new(format!("El corpus no es UTF-8 válido: {}", path.display())))
}

#[derive(Clone, Copy, Debug)]
struct Sequence {
    offset: usize,
    length: usize,
}

#[derive(Clone, Debug)]
pub struct TextDataLoader {
    seq_len: usize,
    batch_size: usize,
    training_tokens: Vec<u32>,
    validation_tokens: Vec<u32>,
    training_sequences: Vec<Sequence>,
    validation_sequences: Vec<Sequence>,
    training_order: Vec<usize>,
    validation_order: Vec<usize>,
    training_cursor: usize,
    validation_cursor: usize,
    training_dropped_tokens: usize,
    validation_dropped_tokens: usize,
    shuffle_training: bool,
    rng: Rng,
}

impl TextDataLoader {
    pub fn new(
        tokenizer: &BpeTokenizer,
        training_text: &str,
        validation_text: &str,
        seq_len: usize,
        batch_size: usize,
        seed: u32,
        shuffle_training: bool,
    ) -> Result<Self> {
        if seq_len == 0 || batch_size == 0 {
            return Err(Error::new(
                "Sequence and batch sizes must be greater than zero",
            ));
        }
        let training_tokens = tokenizer.encode(training_text.as_bytes());
        let validation_tokens = tokenizer.encode(validation_text.as_bytes());
        let (training_sequences, training_dropped_tokens) =
            make_sequences(&training_tokens, seq_len);
        let (validation_sequences, validation_dropped_tokens) =
            make_sequences(&validation_tokens, seq_len);
        let training_order = (0..training_sequences.len()).collect();
        let validation_order = (0..validation_sequences.len()).collect();
        let mut loader = Self {
            seq_len,
            batch_size,
            training_tokens,
            validation_tokens,
            training_sequences,
            validation_sequences,
            training_order,
            validation_order,
            training_cursor: 0,
            validation_cursor: 0,
            training_dropped_tokens,
            validation_dropped_tokens,
            shuffle_training,
            rng: Rng::new(seed),
        };
        loader.reset_epoch();
        loader.reset_validation();
        Ok(loader)
    }

    pub fn reset_epoch(&mut self) {
        self.training_cursor = 0;
        if self.shuffle_training {
            shuffle(&mut self.training_order, &mut self.rng);
        }
    }

    pub fn reset_validation(&mut self) {
        self.validation_cursor = 0;
    }

    pub fn next_batch(&mut self) -> Option<SequenceBatch> {
        make_batch(
            &self.training_sequences,
            &self.training_tokens,
            &self.training_order,
            &mut self.training_cursor,
            self.seq_len,
            self.batch_size,
        )
    }

    pub fn next_validation_batch(&mut self) -> Option<SequenceBatch> {
        make_batch(
            &self.validation_sequences,
            &self.validation_tokens,
            &self.validation_order,
            &mut self.validation_cursor,
            self.seq_len,
            self.batch_size,
        )
    }

    pub fn state(&self) -> DataLoaderState {
        let state = self.rng.state();
        DataLoaderState {
            training_order: self.training_order.clone(),
            validation_order: self.validation_order.clone(),
            training_cursor: self.training_cursor,
            validation_cursor: self.validation_cursor,
            generator_state: format!("{},{},{},{}", state[0], state[1], state[2], state[3]),
        }
    }

    pub fn restore_state(&mut self, state: &DataLoaderState) -> Result<()> {
        let valid_order = |order: &[usize], expected: usize| {
            if order.len() != expected {
                return false;
            }
            let mut seen = vec![false; expected];
            for &index in order {
                if index >= expected || seen[index] {
                    return false;
                }
                seen[index] = true;
            }
            true
        };
        if !valid_order(&state.training_order, self.training_sequences.len())
            || !valid_order(&state.validation_order, self.validation_sequences.len())
            || state.training_cursor > self.training_order.len()
            || state.validation_cursor > self.validation_order.len()
        {
            return Err(Error::new(
                "DataLoader state is incompatible with the current dataset",
            ));
        }

        let parts: Vec<&str> = state.generator_state.split(',').collect();
        if parts.len() != 4 {
            return Err(Error::new("Invalid DataLoader generator state"));
        }
        let mut rng_state = [0u64; 4];
        for (slot, part) in rng_state.iter_mut().zip(parts) {
            *slot = part
                .parse::<u64>()
                .map_err(|_| Error::new("Invalid DataLoader generator state"))?;
        }
        if rng_state.iter().all(|&word| word == 0) {
            return Err(Error::new("Invalid DataLoader generator state"));
        }
        self.rng.restore_state(rng_state);
        self.training_order = state.training_order.clone();
        self.validation_order = state.validation_order.clone();
        self.training_cursor = state.training_cursor;
        self.validation_cursor = state.validation_cursor;
        Ok(())
    }

    pub fn training_sample_count(&self) -> usize {
        self.training_sequences.len()
    }

    pub fn validation_sample_count(&self) -> usize {
        self.validation_sequences.len()
    }

    pub fn training_token_count(&self) -> usize {
        self.training_tokens.len()
    }

    pub fn validation_token_count(&self) -> usize {
        self.validation_tokens.len()
    }

    pub fn training_dropped_tokens(&self) -> usize {
        self.training_dropped_tokens
    }

    pub fn validation_dropped_tokens(&self) -> usize {
        self.validation_dropped_tokens
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

fn make_sequences(tokens: &[u32], seq_len: usize) -> (Vec<Sequence>, usize) {
    let mut sequences = Vec::new();
    let mut dropped_tokens = 0usize;
    let mut offset = 0usize;
    while offset + 1 < tokens.len() {
        let length = seq_len.min(tokens.len() - 1 - offset);
        sequences.push(Sequence { offset, length });
        offset += length;
    }
    if sequences.is_empty() && tokens.len() == 1 {
        dropped_tokens = 1;
    }
    (sequences, dropped_tokens)
}

fn make_batch(
    sequences: &[Sequence],
    tokens: &[u32],
    order: &[usize],
    cursor: &mut usize,
    seq_len: usize,
    batch_size: usize,
) -> Option<SequenceBatch> {
    if *cursor >= order.len() {
        return None;
    }
    let batch_size = batch_size.min(order.len() - *cursor);
    let capacity = batch_size * seq_len;
    let mut batch = SequenceBatch {
        inputs: vec![0; capacity],
        targets: vec![0; capacity],
        mask: vec![0; capacity],
        batch_size,
        seq_len,
        real_tokens: 0,
    };

    for sample in 0..batch_size {
        let sequence = sequences[order[*cursor + sample]];
        for position in 0..sequence.length {
            let destination = sample * seq_len + position;
            batch.inputs[destination] = tokens[sequence.offset + position];
            batch.targets[destination] = tokens[sequence.offset + position + 1];
            batch.mask[destination] = 1;
        }
        batch.real_tokens += sequence.length;
    }
    *cursor += batch_size;
    Some(batch)
}

fn shuffle(values: &mut [usize], rng: &mut Rng) {
    for index in (1..values.len()).rev() {
        let other = rng.below(index + 1);
        values.swap(index, other);
    }
}
