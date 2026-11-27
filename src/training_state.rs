//! Persistencia legible del estado de entrenamiento (`*.data`).

use crate::data::DataLoaderState;
use crate::error::{Error, Result};
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingState {
    pub corpus_path: String,
    pub tokenizer_path: String,
    pub epoch: usize,
    pub processed_tokens: usize,
    pub data: DataLoaderState,
}

pub fn save_training_state(path: &Path, state: &TrainingState) -> Result<()> {
    let mut output = String::new();
    output.push_str("MINI_GPT_TRAINING_STATE 1\n");
    output.push_str(&format!("corpus {}\n", quote(&state.corpus_path)));
    output.push_str(&format!("tokenizer {}\n", quote(&state.tokenizer_path)));
    output.push_str(&format!("epoch {}\n", state.epoch));
    output.push_str(&format!("processed_tokens {}\n", state.processed_tokens));
    output.push_str(&format!(
        "generator {}\n",
        quote(&state.data.generator_state)
    ));
    output.push_str(&format!("training_cursor {}\n", state.data.training_cursor));
    output.push_str(&format!(
        "validation_cursor {}\n",
        state.data.validation_cursor
    ));
    output.push_str("training_order");
    for &index in &state.data.training_order {
        output.push(' ');
        output.push_str(&index.to_string());
    }
    output.push('\n');
    output.push_str("validation_order");
    for &index in &state.data.validation_order {
        output.push(' ');
        output.push_str(&index.to_string());
    }
    output.push('\n');
    std::fs::write(path, output)
        .map_err(|error| Error::new(format!("Failed to write training state: {error}")))
}

pub fn load_training_state(path: &Path) -> Result<TrainingState> {
    let content = std::fs::read_to_string(path).map_err(|error| {
        Error::new(format!(
            "Cannot open training state {}: {error}",
            path.display()
        ))
    })?;
    let mut lines = content.lines();
    let header = lines
        .next()
        .ok_or_else(|| Error::new("Invalid training state header"))?;
    if header.trim() != "MINI_GPT_TRAINING_STATE 1" {
        return Err(Error::new("Invalid training state header"));
    }

    let corpus_path = parse_quoted(next_line(&mut lines, "corpus")?, "corpus")?;
    let tokenizer_path = parse_quoted(next_line(&mut lines, "tokenizer")?, "tokenizer")?;
    let epoch = parse_number(next_line(&mut lines, "epoch")?, "epoch")?;
    let processed_tokens = parse_number(
        next_line(&mut lines, "processed_tokens")?,
        "processed_tokens",
    )?;
    let generator_state = parse_quoted(next_line(&mut lines, "generator")?, "generator")?;
    let training_cursor =
        parse_number(next_line(&mut lines, "training_cursor")?, "training_cursor")?;
    let validation_cursor = parse_number(
        next_line(&mut lines, "validation_cursor")?,
        "validation_cursor",
    )?;
    let training_order = parse_list(next_line(&mut lines, "training_order")?, "training_order")?;
    let validation_order = parse_list(
        next_line(&mut lines, "validation_order")?,
        "validation_order",
    )?;

    Ok(TrainingState {
        corpus_path,
        tokenizer_path,
        epoch,
        processed_tokens,
        data: DataLoaderState {
            training_order,
            validation_order,
            training_cursor,
            validation_cursor,
            generator_state,
        },
    })
}

fn next_line<'a>(lines: &mut impl Iterator<Item = &'a str>, key: &str) -> Result<&'a str> {
    let line = lines
        .next()
        .ok_or_else(|| Error::new(format!("Missing {key} in training state")))?;
    if !line.starts_with(key) {
        return Err(Error::new(format!("Missing {key} in training state")));
    }
    Ok(line[key.len()..].trim())
}

fn parse_quoted(value: &str, key: &str) -> Result<String> {
    let bytes = value.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'"' || bytes[bytes.len() - 1] != b'"' {
        return Err(Error::new(format!("Invalid quoted value for {key}")));
    }
    let inner = &value[1..value.len() - 1];
    let mut output = String::with_capacity(inner.len());
    let mut escaped = false;
    for character in inner.chars() {
        if escaped {
            output.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else {
            output.push(character);
        }
    }
    Ok(output)
}

fn parse_number(value: &str, key: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| Error::new(format!("Invalid number for {key}: {value}")))
}

fn parse_list(value: &str, key: &str) -> Result<Vec<usize>> {
    let mut result = Vec::new();
    for item in value.split_whitespace() {
        result.push(
            item.parse::<usize>()
                .map_err(|_| Error::new(format!("Invalid index in {key}: {item}")))?,
        );
    }
    Ok(result)
}

fn quote(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        if character == '"' || character == '\\' {
            output.push('\\');
        }
        output.push(character);
    }
    output.push('"');
    output
}
