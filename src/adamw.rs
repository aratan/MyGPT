//! Optimizador AdamW con recorte de norma global.

use crate::error::{Error, Result};
use crate::gpt::GPT;
use crate::serialize::{read_file, BytesWriter};
use std::path::Path;

const FILE_MAGIC: &[u8; 8] = b"MINIADAM";
const FILE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct AdamWConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.0003,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            weight_decay: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct OptimizerState {
    first_moment: Vec<f32>,
    second_moment: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct AdamW {
    config: AdamWConfig,
    states: Vec<OptimizerState>,
    weight_decay_flags: Vec<bool>,
    step_count: usize,
}

impl AdamW {
    pub fn new(model: &GPT, config: AdamWConfig) -> Result<Self> {
        let config = validate_config(config)?;
        let parameter_count = model.parameter_count_refs();
        if parameter_count == 0 {
            return Err(Error::new("AdamW requires at least one parameter tensor"));
        }
        let mut states = Vec::with_capacity(parameter_count);
        let mut weight_decay_flags = Vec::with_capacity(parameter_count);
        for index in 0..parameter_count {
            let (parameter, gradient) = model
                .parameter(index)
                .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
            if parameter.shape() != gradient.shape() {
                return Err(Error::new("Invalid parameter/gradient binding in AdamW"));
            }
            let (_, decay) = model
                .parameter_spec(index)
                .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
            states.push(OptimizerState {
                first_moment: vec![0.0; parameter.size()],
                second_moment: vec![0.0; parameter.size()],
            });
            weight_decay_flags.push(decay);
        }
        Ok(Self {
            config,
            states,
            weight_decay_flags,
            step_count: 0,
        })
    }

    pub fn step(&mut self, model: &mut GPT) -> Result<()> {
        self.validate_before_step(model)?;
        if self.step_count == usize::MAX {
            return Err(Error::new("AdamW step counter overflow"));
        }
        self.step_count += 1;

        let beta1 = f64::from(self.config.beta1);
        let beta2 = f64::from(self.config.beta2);
        let learning_rate = f64::from(self.config.learning_rate);
        let epsilon = f64::from(self.config.epsilon);
        let weight_decay = f64::from(self.config.weight_decay);
        let step = self.step_count as f64;
        let first_correction = 1.0 - beta1.powf(step);
        let second_correction = 1.0 - beta2.powf(step);

        for index in 0..self.states.len() {
            let (parameter, gradient) = model
                .parameter_mut(index)
                .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
            let state = &mut self.states[index];
            let decay = self.weight_decay_flags[index];
            let gradient = gradient.data();
            let parameter = parameter.data_mut();
            for i in 0..parameter.len() {
                let gradient = f64::from(gradient[i]);
                let first = beta1 * f64::from(state.first_moment[i]) + (1.0 - beta1) * gradient;
                let second =
                    beta2 * f64::from(state.second_moment[i]) + (1.0 - beta2) * gradient * gradient;
                state.first_moment[i] = first as f32;
                state.second_moment[i] = second as f32;

                let corrected_first = first / first_correction;
                let corrected_second = second / second_correction;
                let value = f64::from(parameter[i]);
                let adam_update =
                    learning_rate * corrected_first / (corrected_second.sqrt() + epsilon);
                let decay_update = if decay {
                    learning_rate * weight_decay * value
                } else {
                    0.0
                };
                parameter[i] = (value - adam_update - decay_update) as f32;
            }
        }
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut writer = BytesWriter::new();
        writer.bytes(FILE_MAGIC);
        writer.u32(FILE_VERSION);
        writer.f32(self.config.learning_rate);
        writer.f32(self.config.beta1);
        writer.f32(self.config.beta2);
        writer.f32(self.config.epsilon);
        writer.f32(self.config.weight_decay);
        writer.u64(self.step_count as u64);
        writer.u64(self.states.len() as u64);
        for (state, &decay) in self.states.iter().zip(&self.weight_decay_flags) {
            writer.u8(u8::from(decay));
            writer.u64(state.first_moment.len() as u64);
            for &value in &state.first_moment {
                if !value.is_finite() {
                    return Err(Error::new("Cannot save non-finite AdamW state"));
                }
                writer.f32(value);
            }
            for &value in &state.second_moment {
                if !value.is_finite() {
                    return Err(Error::new("Cannot save non-finite AdamW state"));
                }
                writer.f32(value);
            }
        }
        writer.finish(path)
    }

    pub fn load(path: &Path, model: &GPT) -> Result<Self> {
        let mut reader = read_file(path)?;
        reader.magic(FILE_MAGIC)?;
        if reader.u32()? != FILE_VERSION {
            return Err(Error::new("Unsupported AdamW file version"));
        }
        let config = AdamWConfig {
            learning_rate: reader.f32()?,
            beta1: reader.f32()?,
            beta2: reader.f32()?,
            epsilon: reader.f32()?,
            weight_decay: reader.f32()?,
        };
        let step_count = reader.u64()? as usize;
        let parameter_count = reader.u64()? as usize;
        if parameter_count != model.parameter_count_refs() {
            return Err(Error::new("AdamW parameter count does not match the model"));
        }

        let mut optimizer = Self::new(model, config)?;
        optimizer.step_count = step_count;
        for index in 0..parameter_count {
            let decay = reader.u8()?;
            if decay > 1 {
                return Err(Error::new("Invalid AdamW weight decay flag"));
            }
            optimizer.weight_decay_flags[index] = decay != 0;
            let value_count = reader.u64()? as usize;
            let (parameter, _) = model
                .parameter(index)
                .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
            if value_count != parameter.size() {
                return Err(Error::new("AdamW state dimensions do not match the model"));
            }
            optimizer.states[index].first_moment = reader.values(value_count)?;
            optimizer.states[index].second_moment = reader.values(value_count)?;
        }
        reader.expect_end()?;
        Ok(optimizer)
    }

    pub fn config(&self) -> &AdamWConfig {
        &self.config
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn parameter_count(&self) -> usize {
        self.states.len()
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    fn validate_before_step(&self, model: &GPT) -> Result<()> {
        if self.states.len() != model.parameter_count_refs() {
            return Err(Error::new("AdamW state does not match parameter count"));
        }
        for index in 0..self.states.len() {
            let (parameter, gradient) = model
                .parameter(index)
                .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
            let (name, _) = model
                .parameter_spec(index)
                .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
            if parameter.shape() != gradient.shape() {
                return Err(Error::new("Invalid parameter/gradient binding in AdamW"));
            }
            if self.states[index].first_moment.len() != parameter.size() {
                return Err(Error::new("AdamW state dimensions do not match parameters"));
            }
            if parameter.data().iter().any(|value| !value.is_finite()) {
                return Err(Error::new(format!("Non-finite parameter in {name}")));
            }
            if gradient.data().iter().any(|value| !value.is_finite()) {
                return Err(Error::new(format!("Non-finite gradient in {name}")));
            }
        }
        Ok(())
    }
}

fn validate_config(config: AdamWConfig) -> Result<AdamWConfig> {
    if !config.learning_rate.is_finite()
        || config.learning_rate < 0.0
        || !config.beta1.is_finite()
        || config.beta1 < 0.0
        || config.beta1 >= 1.0
        || !config.beta2.is_finite()
        || config.beta2 < 0.0
        || config.beta2 >= 1.0
        || !config.epsilon.is_finite()
        || config.epsilon <= 0.0
        || !config.weight_decay.is_finite()
        || config.weight_decay < 0.0
    {
        return Err(Error::new("Invalid AdamW configuration"));
    }
    Ok(config)
}

pub fn gradient_global_norm(model: &GPT) -> Result<f64> {
    let mut squared_sum = 0.0f64;
    for index in 0..model.parameter_count_refs() {
        let (parameter, gradient) = model
            .parameter(index)
            .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
        let (name, _) = model
            .parameter_spec(index)
            .ok_or_else(|| Error::new("Invalid parameter index in AdamW"))?;
        if parameter.shape() != gradient.shape() {
            return Err(Error::new("Invalid parameter/gradient binding in AdamW"));
        }
        for &value in gradient.data() {
            if !value.is_finite() {
                return Err(Error::new(format!("Non-finite gradient in {name}")));
            }
            squared_sum += f64::from(value) * f64::from(value);
        }
    }
    let norm = squared_sum.sqrt();
    if !norm.is_finite() {
        return Err(Error::new("Non-finite global gradient norm"));
    }
    Ok(norm)
}

pub fn clip_gradients(model: &mut GPT, max_norm: f64) -> Result<f64> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(Error::new(
            "Gradient clipping norm must be finite and positive",
        ));
    }
    let norm = gradient_global_norm(model)?;
    if norm > max_norm {
        let scale = (max_norm / norm) as f32;
        for gradient in model.gradients_mut().tensors_mut() {
            for value in gradient.data_mut() {
                *value *= scale;
            }
        }
    }
    Ok(norm)
}
