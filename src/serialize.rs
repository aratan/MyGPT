//! Lectura y escritura binaria compatible con los formatos del original C++
//! (`MINIEMBD`, `MINIATTN`, `MINITRFB`, `MINIGPTM`, `MINIADAM`, `MINIGPTB`).

use crate::error::{Error, Result};
use std::fs;
use std::io::Write;
use std::path::Path;

pub struct BytesWriter {
    buffer: Vec<u8>,
}

impl BytesWriter {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
        }
    }

    pub fn u8(&mut self, value: u8) {
        self.buffer.push(value);
    }

    pub fn u32(&mut self, value: u32) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    pub fn u64(&mut self, value: u64) {
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    pub fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    pub fn bytes(&mut self, value: &[u8]) {
        self.buffer.extend_from_slice(value);
    }

    pub fn tensor(&mut self, tensor: &crate::tensor::Tensor) {
        for &value in tensor.data() {
            self.f32(value);
        }
    }

    pub fn finish(self, path: &Path) -> Result<()> {
        let mut file = fs::File::create(path)
            .map_err(|error| Error::new(format!("Cannot open {}: {error}", path.display())))?;
        file.write_all(&self.buffer)
            .map_err(|error| Error::new(format!("Failed to write {}: {error}", path.display())))?;
        file.flush()?;
        Ok(())
    }
}

impl Default for BytesWriter {
    fn default() -> Self {
        Self::new()
    }
}

pub fn read_file(path: &Path) -> Result<BytesReader> {
    let data = fs::read(path)
        .map_err(|error| Error::new(format!("Cannot open {}: {error}", path.display())))?;
    Ok(BytesReader { data, position: 0 })
}

pub struct BytesReader {
    data: Vec<u8>,
    position: usize,
}

impl BytesReader {
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data, position: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    pub fn u8(&mut self) -> Result<u8> {
        if self.remaining() < 1 {
            return Err(Error::new("Unexpected end of file"));
        }
        let value = self.data[self.position];
        self.position += 1;
        Ok(value)
    }

    pub fn u32(&mut self) -> Result<u32> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub fn u64(&mut self) -> Result<u64> {
        let bytes = self.bytes(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(array))
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub fn bytes(&mut self, length: usize) -> Result<&[u8]> {
        if self.remaining() < length {
            return Err(Error::new("Unexpected end of file"));
        }
        let start = self.position;
        self.position += length;
        Ok(&self.data[start..start + length])
    }

    pub fn magic(&mut self, expected: &[u8]) -> Result<()> {
        let actual = self.bytes(expected.len())?;
        if actual != expected {
            return Err(Error::new("Invalid file magic"));
        }
        Ok(())
    }

    pub fn dimension(&mut self) -> Result<usize> {
        let value = self.u64()?;
        if value == 0 || value > usize::MAX as u64 {
            return Err(Error::new("Invalid dimension in file"));
        }
        Ok(value as usize)
    }

    pub fn values(&mut self, count: usize) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.f32()?);
        }
        Ok(values)
    }

    pub fn expect_end(&self) -> Result<()> {
        if self.remaining() != 0 {
            return Err(Error::new("Unexpected trailing data in file"));
        }
        Ok(())
    }
}
