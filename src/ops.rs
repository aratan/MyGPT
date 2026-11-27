//! Operaciones compartidas: proyección lineal 3D y GELU exacta.

use crate::error::{Error, Result};
use crate::tensor::{linear_flat, Tensor};

/// `input [batch, seq, input_dim] · weights [input_dim, output_dim]`.
pub fn linear(input: &Tensor, weights: &Tensor) -> Result<Tensor> {
    if input.rank() != 3 || weights.rank() != 2 {
        return Err(Error::new("Linear expects a 3D input and a 2D weight matrix"));
    }
    let batch_size = input.shape()[0];
    let seq_len = input.shape()[1];
    let input_dim = input.shape()[2];
    let output_dim = weights.shape()[1];
    if weights.shape()[0] != input_dim {
        return Err(Error::new("Incompatible shapes in linear operation"));
    }

    let rows = batch_size * seq_len;
    let flattened = linear_flat(input.data(), weights, rows, input_dim, output_dim)?;
    Tensor::new(vec![batch_size, seq_len, output_dim], flattened.into_data())
}

/// GELU exacta: `0.5 · x · (1 + erf(x / sqrt(2)))`.
pub fn gelu(input: &Tensor) -> Tensor {
    const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;
    input.map(|value| {
        let value = f64::from(value);
        let activated = 0.5 * value * (1.0 + erf(value * SQRT_HALF));
        activated as f32
    })
}

/// Derivada de la GELU exacta.
pub fn gelu_backward(input: &Tensor, output_gradient: &Tensor) -> Result<Tensor> {
    if input.shape() != output_gradient.shape() {
        return Err(Error::new("GELU backward requires matching shapes"));
    }
    const SQRT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const INVERSE_SQRT_TWO_PI: f64 = 0.398_942_280_401_432_7;
    let mut gradient = vec![0.0f32; input.size()];
    for (index, target) in gradient.iter_mut().enumerate() {
        let value = f64::from(input.data()[index]);
        let cdf = 0.5 * (1.0 + erf(value * SQRT_HALF));
        let pdf = INVERSE_SQRT_TWO_PI * (-0.5 * value * value).exp();
        *target = (f64::from(output_gradient.data()[index]) * (cdf + value * pdf)) as f32;
    }
    Ok(Tensor::new(input.shape().to_vec(), gradient)?)
}

/// Aproximación de `erf` con precisión cercana a la de la libm (Abramowitz–Stegun 7.1.26).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t
        - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}
