//! Lectura y escritura binaria compatible con los formatos del original C++
//! (`MINIEMBD`, `MINIATTN`, `MINITRFB`, `MINIGPTM`, `MINIADAM`, `MINIGPTB`).

use crate::error::{Error, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Replaces a file only after the complete new contents have reached disk.
///
/// The temporary file is created beside the destination, so the final rename
/// stays on the same filesystem. The destination remains untouched if writing
/// or syncing the temporary file fails.
pub(crate) fn write_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| Error::new(format!("Invalid output path: {}", path.display())))?;

    let (temporary_path, mut file) = create_temporary_file(parent, file_name)?;
    let write_result = file
        .write_all(contents)
        .and_then(|()| file.sync_all());
    drop(file);

    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(Error::new(format!(
            "Failed to write {}: {error}",
            path.display()
        )));
    }

    if let Err(error) = fs::rename(&temporary_path, path) {
        let _ = fs::remove_file(&temporary_path);
        return Err(Error::new(format!(
            "Cannot replace {}: {error}",
            path.display()
        )));
    }
    Ok(())
}

fn create_temporary_file(
    parent: &Path,
    file_name: &std::ffi::OsStr,
) -> Result<(PathBuf, fs::File)> {
    for _ in 0..100 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = std::ffi::OsString::from(".");
        temporary_name.push(file_name);
        temporary_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
        let temporary_path = parent.join(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(Error::new(format!(
                    "Cannot create temporary file beside {}: {error}",
                    parent.display()
                )))
            }
        }
    }
    Err(Error::new(format!(
        "Cannot allocate a unique temporary file beside {}",
        parent.display()
    )))
}

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
        write_atomic(path, &self.buffer)
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
        self.data.len().saturating_sub(self.position)
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
        if count > self.remaining() / std::mem::size_of::<f32>() {
            return Err(Error::new("Unexpected end of file while reading values"));
        }
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
