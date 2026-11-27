//! Mini GPT en Rust: tensor, BPE, transformer, entrenamiento e inferencia.
//!
//! Port del proyecto C++ homónimo, con los mismos formatos binarios de
//! checkpoint y una implementación sin dependencias externas.

pub mod attention;
pub mod backprop;
pub mod chat_mode;
pub mod cli;
pub mod data;
pub mod embedding;
pub mod error;
pub mod gpt;
pub mod inference;
pub mod layer_norm;
pub mod loss;
pub mod ops;
pub mod rng;
pub mod serialize;
pub mod synthetic;
pub mod tensor;
pub mod tokenizer;
pub mod trace;
pub mod training_state;
pub mod transformer;

pub use attention::CausalSelfAttention;
pub use backprop::GPTGradients;
pub use data::{CorpusSplit, DataLoaderState, SequenceBatch, TextDataLoader};
pub use embedding::Embedding;
pub use error::{Error, Result};
pub use gpt::{GPTParameterCount, GenerationConfig, SamplingStrategy, StopCondition, GPT};
pub use layer_norm::LayerNorm;
pub use loss::{cross_entropy, cross_entropy_backward, perplexity, shift_tokens};
pub use tensor::Tensor;
pub use tokenizer::BpeTokenizer;
pub use training_state::{load_training_state, save_training_state, TrainingState};
pub use transformer::TransformerBlock;

pub mod optim {
    pub use crate::adamw::{clip_gradients, gradient_global_norm, AdamW, AdamWConfig};
}

pub mod adamw;
