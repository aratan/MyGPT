//! Embeddings de token y de posición.

use crate::error::{Error, Result};
use crate::serialize::{read_file, BytesWriter};
use crate::tensor::Tensor;
use std::path::Path;

const INITIALIZATION_SCALE: f32 = 0.02;
const FILE_MAGIC: &[u8; 8] = b"MINIEMBD";
const FILE_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
    max_seq_len: usize,
    token_embeddings: Tensor,
    positional_embeddings: Tensor,
}

impl Embedding {
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        max_seq_len: usize,
        seed: u32,
    ) -> Result<Self> {
        for (value, name) in [
            (vocab_size, "vocab_size"),
            (embedding_dim, "embedding_dim"),
            (max_seq_len, "max_seq_len"),
        ] {
            if value == 0 {
                return Err(Error::new(format!("{name} must be greater than zero")));
            }
        }
        Ok(Self {
            vocab_size,
            embedding_dim,
            max_seq_len,
            token_embeddings: Tensor::random(
                [vocab_size, embedding_dim],
                seed,
                -INITIALIZATION_SCALE,
                INITIALIZATION_SCALE,
            )?,
            positional_embeddings: Tensor::random(
                [max_seq_len, embedding_dim],
                seed.wrapping_add(1),
                -INITIALIZATION_SCALE,
                INITIALIZATION_SCALE,
            )?,
        })
    }

    pub fn from_tensors(token_embeddings: Tensor, positional_embeddings: Tensor) -> Result<Self> {
        if token_embeddings.rank() != 2
            || positional_embeddings.rank() != 2
            || token_embeddings.shape()[1] != positional_embeddings.shape()[1]
            || token_embeddings.shape()[0] == 0
            || token_embeddings.shape()[1] == 0
            || positional_embeddings.shape()[0] == 0
        {
            return Err(Error::new("Invalid embedding table shapes"));
        }
        Ok(Self {
            vocab_size: token_embeddings.shape()[0],
            embedding_dim: token_embeddings.shape()[1],
            max_seq_len: positional_embeddings.shape()[0],
            token_embeddings,
            positional_embeddings,
        })
    }

    pub fn token_forward(
        &self,
        token_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.validate_shapes()?;
        self.validate_batch(token_ids.len(), batch_size, seq_len)?;

        let dimension = self.embedding_dim;
        let mut output = vec![0.0f32; batch_size * seq_len * dimension];
        let weights = self.token_embeddings.data();
        for (index, &id) in token_ids.iter().enumerate() {
            if id as usize >= self.vocab_size {
                return Err(Error::new("Token ID is outside the embedding vocabulary"));
            }
            let source = id as usize * dimension;
            let destination = index * dimension;
            output[destination..destination + dimension]
                .copy_from_slice(&weights[source..source + dimension]);
        }
        Tensor::new(vec![batch_size, seq_len, dimension], output)
    }

    pub fn forward(&self, token_ids: &[u32], batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let mut output = self.token_forward(token_ids, batch_size, seq_len)?;
        let dimension = self.embedding_dim;
        let positions = self.positional_embeddings.data();
        let data = output.data_mut();
        for batch in 0..batch_size {
            for position in 0..seq_len {
                let output_offset = (batch * seq_len + position) * dimension;
                let position_offset = position * dimension;
                for index in 0..dimension {
                    data[output_offset + index] += positions[position_offset + index];
                }
            }
        }
        Ok(output)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate_shapes()?;
        let mut writer = BytesWriter::new();
        writer.bytes(FILE_MAGIC);
        writer.u32(FILE_VERSION);
        writer.u64(self.vocab_size as u64);
        writer.u64(self.embedding_dim as u64);
        writer.u64(self.max_seq_len as u64);
        writer.tensor(&self.token_embeddings);
        writer.tensor(&self.positional_embeddings);
        writer.finish(path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = read_file(path)?;
        reader.magic(FILE_MAGIC)?;
        if reader.u32()? != FILE_VERSION {
            return Err(Error::new("Unsupported embedding file version"));
        }
        let vocab_size = reader.dimension()?;
        let embedding_dim = reader.dimension()?;
        let max_seq_len = reader.dimension()?;
        let token_count = vocab_size
            .checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("Embedding dimensions exceed usize range"))?;
        let position_count = max_seq_len
            .checked_mul(embedding_dim)
            .ok_or_else(|| Error::new("Embedding dimensions exceed usize range"))?;
        let expected = (token_count + position_count)
            .checked_mul(4)
            .ok_or_else(|| Error::new("Embedding data size exceeds file format range"))?;
        if reader.remaining() != expected {
            return Err(Error::new("Embedding dimensions do not match file size"));
        }
        let token_embeddings =
            Tensor::new([vocab_size, embedding_dim], reader.values(token_count)?)?;
        let positional_embeddings =
            Tensor::new([max_seq_len, embedding_dim], reader.values(position_count)?)?;
        Self::from_tensors(token_embeddings, positional_embeddings)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn token_embeddings(&self) -> &Tensor {
        &self.token_embeddings
    }

    pub fn token_embeddings_mut(&mut self) -> &mut Tensor {
        &mut self.token_embeddings
    }

    pub fn positional_embeddings(&self) -> &Tensor {
        &self.positional_embeddings
    }

    pub fn positional_embeddings_mut(&mut self) -> &mut Tensor {
        &mut self.positional_embeddings
    }

    fn validate_batch(&self, token_count: usize, batch_size: usize, seq_len: usize) -> Result<()> {
        if seq_len > self.max_seq_len {
            return Err(Error::new("Sequence length exceeds max_seq_len"));
        }
        let expected = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| Error::new("Batch dimensions exceed usize range"))?;
        if token_count != expected {
            return Err(Error::new(
                "Token count does not match batch_size * seq_len",
            ));
        }
        Ok(())
    }

    fn validate_shapes(&self) -> Result<()> {
        if self.token_embeddings.shape() != [self.vocab_size, self.embedding_dim]
            || self.positional_embeddings.shape() != [self.max_seq_len, self.embedding_dim]
        {
            return Err(Error::new(
                "Embedding table shapes do not match the layer configuration",
            ));
        }
        Ok(())
    }
}
