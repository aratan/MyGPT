//! Generador determinista rápido (xoshiro256**), sin dependencias.
//!
//! Sustituye a `std::mt19937`: los valores no son idénticos a los del original
//! de C++, pero toda la inicialización es determinista para una semilla dada.

#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        let mut initial = u64::from(seed) ^ 0x9E37_79B9_7F4A_7C15;
        let mut state = [0u64; 4];
        for word in &mut state {
            *word = splitmix64(&mut initial);
        }
        // xoshiro exige un estado distinto de cero.
        if state.iter().all(|&word| word == 0) {
            state[0] = 1;
        }
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let temporary = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Uniforme en `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits de mantisa: (0, 1) en coma flotante de precisión simple.
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    /// Uniforme en `[min, max)`.
    #[inline]
    pub fn uniform(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    /// Entero en `[0, bound)` sin sesgo apreciable.
    #[inline]
    pub fn below(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    pub fn state(&self) -> [u64; 4] {
        self.state
    }

    pub fn restore_state(&mut self, state: [u64; 4]) {
        self.state = state;
    }
}
