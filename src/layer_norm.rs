//! LayerNorm sobre la última dimensión.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct LayerNorm {
    normalized_dim: usize,
    epsilon: f32,
    gamma: Tensor,
    beta: Tensor,
}

impl LayerNorm {
    pub fn new(normalized_dim: usize, epsilon: f32) -> Result<Self> {
        if normalized_dim == 0 {
            return Err(Error::new("normalized_dim must be greater than zero"));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::new(
                "LayerNorm epsilon must be finite and greater than zero",
            ));
        }
        Ok(Self {
            normalized_dim,
            epsilon,
            gamma: Tensor::ones([normalized_dim]),
            beta: Tensor::zeros([normalized_dim]),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.validate_parameters()?;
        if input.rank() == 0 || input.shape().last() != Some(&self.normalized_dim) {
            return Err(Error::new("LayerNorm input must end with normalized_dim"));
        }

        let dimension = self.normalized_dim;
        let rows = input.size() / dimension;
        let input_data = input.data();
        let gamma = self.gamma.data();
        let beta = self.beta.data();
        let mut output = vec![0.0f32; input.size()];
        let inverse_dimension = 1.0f64 / dimension as f64;

        for row in 0..rows {
            let offset = row * dimension;
            let values = &input_data[offset..offset + dimension];
            let mut sum = 0.0f64;
            for &value in values {
                sum += f64::from(value);
            }
            let mean = sum * inverse_dimension;

            let mut squared_differences = 0.0f64;
            for &value in values {
                let difference = f64::from(value) - mean;
                squared_differences += difference * difference;
            }
            let variance = squared_differences * inverse_dimension;
            let inverse_standard_deviation = 1.0 / (variance + f64::from(self.epsilon)).sqrt();

            let out_row = &mut output[offset..offset + dimension];
            for index in 0..dimension {
                let normalized = (f64::from(values[index]) - mean) * inverse_standard_deviation;
                out_row[index] =
                    (normalized * f64::from(gamma[index]) + f64::from(beta[index])) as f32;
            }
        }

        Tensor::new(input.shape().to_vec(), output)
    }

    pub fn normalized_dim(&self) -> usize {
        self.normalized_dim
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut Tensor {
        &mut self.gamma
    }

    pub fn beta(&self) -> &Tensor {
        &self.beta
    }

    pub fn beta_mut(&mut self) -> &mut Tensor {
        &mut self.beta
    }

    pub fn set_parameters(&mut self, gamma: Tensor, beta: Tensor) -> Result<()> {
        if gamma.shape() != [self.normalized_dim] || beta.shape() != [self.normalized_dim] {
            return Err(Error::new(
                "LayerNorm parameter shapes do not match normalized_dim",
            ));
        }
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }

    fn validate_parameters(&self) -> Result<()> {
        if self.gamma.shape() != [self.normalized_dim] || self.beta.shape() != [self.normalized_dim]
        {
            return Err(Error::new(
                "LayerNorm parameter shapes do not match normalized_dim",
            ));
        }
        Ok(())
    }
}
