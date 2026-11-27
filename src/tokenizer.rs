//! Byte-level BPE con tokens especiales, compatible con `MINIGPTB`.

use crate::error::{Error, Result};
use crate::serialize::{read_file, BytesWriter};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

pub const USER_ID: u32 = 256;
pub const ASSISTANT_ID: u32 = 257;
pub const EOT_ID: u32 = 258;
pub const END_ID: u32 = 259;
pub const BASE_VOCABULARY_SIZE: usize = 260;

pub const SPECIAL_TOKENS: [&[u8]; 4] = [b"<|user|>", b"<|assistant|>", b"<|eot|>", b"<|end|>"];

const FILE_MAGIC: &[u8; 8] = b"MINIGPTB";
const FILE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
}

/// Hasher rápido (estilo FxHash) para las tablas de pares.
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut array = [0u8; 8];
            array.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(array));
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut array = [0u8; 8];
            array[..remainder.len()].copy_from_slice(remainder);
            self.add(u64::from_le_bytes(array));
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    max_vocab_size: usize,
    vocabulary: Vec<Vec<u8>>,
    merges: Vec<Merge>,
    merge_ranks: FastMap<(u32, u32), usize>,
}

impl BpeTokenizer {
    pub fn new(max_vocab_size: usize) -> Result<Self> {
        if !(BASE_VOCABULARY_SIZE..=u32::MAX as usize).contains(&max_vocab_size) {
            return Err(Error::new(
                "Vocabulary size must be between 260 and uint32_t max",
            ));
        }
        Ok(Self {
            max_vocab_size,
            vocabulary: base_vocabulary(),
            merges: Vec::new(),
            merge_ranks: FastMap::default(),
        })
    }

    /// Entrena con actualización incremental de frecuencias de pares.
    pub fn train(&mut self, corpus: &[u8], mut progress: impl FnMut(usize, usize)) {
        self.reset_to_base();
        if corpus.is_empty() || self.max_vocab_size == BASE_VOCABULARY_SIZE {
            return;
        }

        let mut sequences: Vec<Vec<u32>> = Vec::new();
        for segment in split_specials(corpus) {
            if let Segment::Normal(normal) = segment {
                if !normal.is_empty() {
                    sequences.push(byte_sequence(normal));
                }
            }
        }

        let target = self.max_vocab_size - BASE_VOCABULARY_SIZE;
        let mut counts: FastMap<(u32, u32), u64> = FastMap::default();
        for sequence in &sequences {
            for pair in sequence.windows(2) {
                *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
            }
        }

        while self.vocabulary.len() < self.max_vocab_size {
            // Mejor par: máxima frecuencia y, a igualdad, el menor par.
            let mut best: Option<((u32, u32), u64)> = None;
            for (&pair, &count) in &counts {
                if count == 0 {
                    continue;
                }
                match best {
                    None => best = Some((pair, count)),
                    Some((best_pair, best_count)) => {
                        if count > best_count || (count == best_count && pair < best_pair) {
                            best = Some((pair, count));
                        }
                    }
                }
            }
            let Some((best_pair, _)) = best else { break };

            let merge = Merge {
                left: best_pair.0,
                right: best_pair.1,
            };
            let result_id = self.vocabulary.len() as u32;
            let mut merged = self.vocabulary[merge.left as usize].clone();
            merged.extend_from_slice(&self.vocabulary[merge.right as usize]);
            self.vocabulary.push(merged);
            self.merge_ranks
                .insert((merge.left, merge.right), self.merges.len());
            self.merges.push(merge);

            for sequence in &mut sequences {
                // Quita los pares actuales, fusiona y vuelve a contar.
                for pair in sequence.windows(2) {
                    if let Some(count) = counts.get_mut(&(pair[0], pair[1])) {
                        *count -= 1;
                    }
                }
                apply_merge(sequence, merge, result_id);
                for pair in sequence.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
                }
            }
            progress(self.merges.len(), target);
        }
    }

    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let mut output = Vec::with_capacity(text.len());
        for segment in split_specials(text) {
            match segment {
                Segment::Normal(normal) => self.append_encoded(normal, &mut output),
                Segment::Special(special) => output.push(special),
            }
        }
        output
    }

    pub fn decode(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(ids.len());
        for &id in ids {
            let token = self
                .vocabulary
                .get(id as usize)
                .ok_or_else(|| Error::new("Token ID is out of range"))?;
            output.extend_from_slice(token);
        }
        Ok(output)
    }

    pub fn decode_string(&self, ids: &[u32]) -> Result<String> {
        Ok(String::from_utf8(self.decode(ids)?)?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BytesWriter::new();
        writer.bytes(FILE_MAGIC);
        writer.u32(FILE_VERSION);
        writer.u32(self.max_vocab_size as u32);
        writer.u32(self.vocabulary.len() as u32);
        writer.u32(self.merges.len() as u32);
        for token in &self.vocabulary {
            writer.u64(token.len() as u64);
            writer.bytes(token);
        }
        for merge in &self.merges {
            writer.u32(merge.left);
            writer.u32(merge.right);
        }
        writer.finish(path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut reader = read_file(path)?;
        reader.magic(FILE_MAGIC)?;
        if reader.u32()? != FILE_VERSION {
            return Err(Error::new("Unsupported tokenizer file version"));
        }
        let max_vocab_size = reader.u32()? as usize;
        let vocab_count = reader.u32()? as usize;
        let merge_count = reader.u32()? as usize;
        if vocab_count < BASE_VOCABULARY_SIZE
            || vocab_count > max_vocab_size
            || merge_count != vocab_count - BASE_VOCABULARY_SIZE
        {
            return Err(Error::new("Invalid tokenizer vocabulary counts"));
        }

        let mut tokenizer = Self::new(max_vocab_size)?;
        for index in 0..vocab_count {
            let length = reader.u64()? as usize;
            if length > reader.remaining() {
                return Err(Error::new("Invalid token length in tokenizer file"));
            }
            let token = reader.bytes(length)?.to_vec();
            if index < BASE_VOCABULARY_SIZE {
                if token != tokenizer.vocabulary[index] {
                    return Err(Error::new(
                        "Tokenizer base vocabulary does not match its IDs",
                    ));
                }
            } else {
                tokenizer.vocabulary.push(token);
            }
        }

        for index in 0..merge_count {
            let merge = Merge {
                left: reader.u32()?,
                right: reader.u32()?,
            };
            let result_id = (BASE_VOCABULARY_SIZE + index) as u32;
            if merge.left >= result_id
                || merge.right >= result_id
                || is_special(merge.left)
                || is_special(merge.right)
            {
                return Err(Error::new("Invalid merge in tokenizer file"));
            }
            let mut expected = tokenizer.vocabulary[merge.left as usize].clone();
            expected.extend_from_slice(&tokenizer.vocabulary[merge.right as usize]);
            if tokenizer.vocabulary[result_id as usize] != expected {
                return Err(Error::new(
                    "Tokenizer merge and vocabulary are inconsistent",
                ));
            }
            tokenizer.merges.push(merge);
            if tokenizer
                .merge_ranks
                .insert((merge.left, merge.right), index)
                .is_some()
            {
                return Err(Error::new("Duplicate merge in tokenizer file"));
            }
        }
        reader.expect_end()?;
        Ok(tokenizer)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn max_vocab_size(&self) -> usize {
        self.max_vocab_size
    }

    pub fn vocabulary(&self) -> &[Vec<u8>] {
        &self.vocabulary
    }

    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    pub fn token(&self, id: u32) -> Result<&[u8]> {
        self.vocabulary
            .get(id as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::new("Token ID is out of range"))
    }

    fn reset_to_base(&mut self) {
        self.vocabulary.truncate(BASE_VOCABULARY_SIZE);
        self.merges.clear();
        self.merge_ranks.clear();
    }

    fn append_encoded(&self, text: &[u8], output: &mut Vec<u32>) {
        let mut sequence = byte_sequence(text);
        while sequence.len() >= 2 {
            let mut best_rank = self.merges.len();
            let mut best_pair = (0u32, 0u32);
            for pair in sequence.windows(2) {
                let key = (pair[0], pair[1]);
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = key;
                    }
                }
            }
            if best_rank == self.merges.len() {
                break;
            }
            apply_merge(
                &mut sequence,
                Merge {
                    left: best_pair.0,
                    right: best_pair.1,
                },
                (BASE_VOCABULARY_SIZE + best_rank) as u32,
            );
        }
        output.extend_from_slice(&sequence);
    }
}

pub fn is_special(id: u32) -> bool {
    (USER_ID..=END_ID).contains(&id)
}

fn base_vocabulary() -> Vec<Vec<u8>> {
    let mut vocabulary: Vec<Vec<u8>> = (0u16..256).map(|value| vec![value as u8]).collect();
    for special in SPECIAL_TOKENS {
        vocabulary.push(special.to_vec());
    }
    vocabulary
}

enum Segment<'a> {
    Normal(&'a [u8]),
    Special(u32),
}

fn split_specials(text: &[u8]) -> Vec<Segment<'_>> {
    let mut segments = Vec::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let mut next_position = usize::MAX;
        let mut next_special = 0usize;
        for (index, special) in SPECIAL_TOKENS.iter().enumerate() {
            if let Some(relative) = find(&text[cursor..], special) {
                let position = cursor + relative;
                if position < next_position {
                    next_position = position;
                    next_special = index;
                }
            }
        }

        if next_position == usize::MAX {
            segments.push(Segment::Normal(&text[cursor..]));
            break;
        }
        if next_position > cursor {
            segments.push(Segment::Normal(&text[cursor..next_position]));
        }
        segments.push(Segment::Special(USER_ID + next_special as u32));
        cursor = next_position + SPECIAL_TOKENS[next_special].len();
    }
    segments
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn byte_sequence(text: &[u8]) -> Vec<u32> {
    text.iter().map(|&byte| u32::from(byte)).collect()
}

fn apply_merge(sequence: &mut Vec<u32>, merge: Merge, result_id: u32) {
    let mut read = 0usize;
    let mut write = 0usize;
    while read < sequence.len() {
        if read + 1 < sequence.len()
            && sequence[read] == merge.left
            && sequence[read + 1] == merge.right
        {
            sequence[write] = result_id;
            write += 1;
            read += 2;
        } else {
            sequence[write] = sequence[read];
            write += 1;
            read += 1;
        }
    }
    sequence.truncate(write);
}
