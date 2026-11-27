//! Modos de conversación y construcción del prompt.

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatMode {
    Chat,
    Completion,
}

pub fn parse_chat_mode(mode: &str) -> Result<ChatMode> {
    match mode {
        "chat" => Ok(ChatMode::Chat),
        "completion" | "continue" => Ok(ChatMode::Completion),
        _ => Err(Error::new("--mode debe ser chat o completion")),
    }
}

pub fn build_chat_prompt(mode: ChatMode, input: &str) -> String {
    match mode {
        ChatMode::Chat => format!("<|user|> {input} <|eot|>\n<|assistant|>"),
        ChatMode::Completion => input.to_string(),
    }
}
