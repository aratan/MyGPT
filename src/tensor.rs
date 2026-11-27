//! Tensor denso en fila mayor, con matmul paralela y sin dependencias.

use crate::error::{Error, Result};
use crate::rng::Rng;

/// Umbral de trabajo (multiplicaciones-acumulaciones) a partir del cual se
/// reparte el matmul entre varios hilos.
const PARALLEL_FLOP_THRESHOLD: usize = 1 << 20;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        let size = element_count(&shape)?;
        if data.len() != size {
            return Err(Error::new(format!(
                "Data size does not match shape {}: expected {}, got {}",
                shape_string(&shape),
                size,
                data.len()
            )));
        }
        let strides = calculate_strides(&shape)?;
        Ok(Self {
            shape,
            strides,
            data,
        })
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let size = element_count(&shape).unwrap_or(0);
        let strides = calculate_strides(&shape).unwrap_or_default();
        Self {
            shape,
            strides,
            data: vec![0.0; size],
        }
    }

    pub fn ones(shape: impl Into<Vec<usize>>) -> Self {
        let mut tensor = Self::zeros(shape);
        tensor.data.fill(1.0);
        tensor
    }

    /// Valores uniformes en `[min, max)` con la semilla indicada.
    pub fn random(shape: impl Into<Vec<usize>>, seed: u32, min: f32, max: f32) -> Result<Self> {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(Error::new(
                "Random bounds must be finite and satisfy min <= max",
            ));
        }
        let mut tensor = Self::zeros(shape);
        let mut rng = Rng::new(seed);
        for value in &mut tensor.data {
            *value = rng.uniform(min, max);
        }
        Ok(tensor)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn dim(&self, axis: usize) -> usize {
        self.shape[axis]
    }

    pub fn at(&self, indices: &[usize]) -> Result<f32> {
        let offset = self.offset(indices)?;
        Ok(self.data[offset])
    }

    pub fn set_at(&mut self, indices: &[usize], value: f32) -> Result<()> {
        let offset = self.offset(indices)?;
        self.data[offset] = value;
        Ok(())
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.elementwise(other, |lhs, rhs| lhs + rhs)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.elementwise(other, |lhs, rhs| lhs - rhs)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.elementwise(other, |lhs, rhs| lhs * rhs)
    }

    fn elementwise(&self, other: &Tensor, operation: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(Error::new(format!(
                "Elementwise operations require identical shapes: {} and {}",
                shape_string(&self.shape),
                shape_string(&other.shape)
            )));
        }
        let mut result = Vec::with_capacity(self.data.len());
        result.extend(
            self.data
                .iter()
                .zip(&other.data)
                .map(|(&lhs, &rhs)| operation(lhs, rhs)),
        );
        Ok(Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: result,
        })
    }

    /// Producto matricial 2D. Reparte las filas del resultado entre hilos y
    /// recorre la matriz derecha en orden de filas para vectorizar el bucle.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if self.rank() != 2 || other.rank() != 2 {
            return Err(Error::new("matmul requires two 2D tensors"));
        }
        let rows = self.shape[0];
        let inner = self.shape[1];
        let columns = other.shape[1];
        if inner != other.shape[0] {
            return Err(Error::new(format!(
                "matmul inner dimensions must match: {} and {}",
                shape_string(&self.shape),
                shape_string(&other.shape)
            )));
        }

        let mut output = vec![0.0f32; rows * columns];
        matmul_into(&self.data, &other.data, &mut output, rows, inner, columns);
        Tensor::new(vec![rows, columns], output)
    }

    pub fn transpose(&self) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(Error::new("transpose requires a 2D tensor"));
        }
        let rows = self.shape[0];
        let columns = self.shape[1];
        let mut output = vec![0.0f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                output[column * rows + row] = self.data[row * columns + column];
            }
        }
        Tensor::new(vec![columns, rows], output)
    }

    /// Cambia la forma in situ, sin copiar ni reordenar los datos.
    pub fn reshape(&mut self, new_shape: impl Into<Vec<usize>>) -> Result<()> {
        let new_shape = new_shape.into();
        let new_size = element_count(&new_shape)?;
        if new_size != self.size() {
            return Err(Error::new(format!(
                "reshape cannot change the element count from {} to {}",
                self.size(),
                new_size
            )));
        }
        self.strides = calculate_strides(&new_shape)?;
        self.shape = new_shape;
        Ok(())
    }

    /// Aplica una función a cada elemento.
    pub fn map(&self, operation: impl Fn(f32) -> f32) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            data: self.data.iter().map(|&value| operation(value)).collect(),
        }
    }

    pub fn scaled(&self, factor: f32) -> Tensor {
        self.map(|value| value * factor)
    }

    /// `out += term`.
    pub fn add_assign(&mut self, term: &Tensor) -> Result<()> {
        if self.shape != term.shape {
            return Err(Error::new(
                "Gradient shapes do not match during accumulation",
            ));
        }
        for (target, &source) in self.data.iter_mut().zip(&term.data) {
            *target += source;
        }
        Ok(())
    }

    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }

    fn offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.rank() {
            return Err(Error::new(format!(
                "Expected {} indices, got {}",
                self.rank(),
                indices.len()
            )));
        }
        let mut offset = 0;
        for (axis, &index) in indices.iter().enumerate() {
            if index >= self.shape[axis] {
                return Err(Error::new(format!(
                    "Index {} is out of range for axis {} with size {}",
                    index, axis, self.shape[axis]
                )));
            }
            offset += index * self.strides[axis];
        }
        Ok(offset)
    }
}

impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

// `&Tensor * &Tensor` también es producto elemento a elemento, como en C++.
impl std::ops::Mul for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

fn matmul_into(
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    rows: usize,
    inner: usize,
    columns: usize,
) {
    let work = rows.saturating_mul(inner).saturating_mul(columns);
    let threads = parallel_threads(work, rows);
    if threads <= 1 {
        matmul_rows(lhs, rhs, output, inner, columns);
        return;
    }
    let rows_per_thread = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in output.chunks_mut(rows_per_thread * columns).enumerate() {
            let first_row = chunk_index * rows_per_thread;
            scope.spawn(move || {
                matmul_rows(&lhs[first_row * inner..], rhs, chunk, inner, columns);
            });
        }
    });
}

#[inline]
fn matmul_rows(lhs: &[f32], rhs: &[f32], output: &mut [f32], inner: usize, columns: usize) {
    for (row, out_row) in output.chunks_mut(columns).enumerate() {
        let lhs_row = &lhs[row * inner..row * inner + inner];
        out_row.fill(0.0);
        for (i, &value) in lhs_row.iter().enumerate() {
            let rhs_row = &rhs[i * columns..i * columns + columns];
            for (out, &rhs_value) in out_row.iter_mut().zip(rhs_row) {
                *out += value * rhs_value;
            }
        }
    }
}

/// Multiplica un lote `[rows, input_dim]` (sin materializar el tensor 2D)
/// por `weights [input_dim, output_dim]`. Devuelve `[rows, output_dim]`.
pub fn linear_flat(
    input: &[f32],
    weights: &Tensor,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<Tensor> {
    if weights.rank() != 2 || weights.shape()[0] != input_dim || weights.shape()[1] != output_dim {
        return Err(Error::new("Incompatible shapes in linear operation"));
    }
    if input.len() != rows * input_dim {
        return Err(Error::new("Input length does not match rows * input_dim"));
    }

    let work = rows.saturating_mul(input_dim).saturating_mul(output_dim);
    let weight_data = weights.data();
    let mut output = vec![0.0f32; rows * output_dim];
    let threads = parallel_threads(work, rows);
    let compute = |first_row: usize, chunk: &mut [f32]| {
        for (local_row, out_row) in chunk.chunks_mut(output_dim).enumerate() {
            let row = first_row + local_row;
            let input_row = &input[row * input_dim..row * input_dim + input_dim];
            out_row.fill(0.0);
            for (i, &value) in input_row.iter().enumerate() {
                let weight_row = &weight_data[i * output_dim..i * output_dim + output_dim];
                for (out, &weight) in out_row.iter_mut().zip(weight_row) {
                    *out += value * weight;
                }
            }
        }
    };

    if threads <= 1 {
        compute(0, &mut output);
    } else {
        let rows_per_thread = rows.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, chunk) in output.chunks_mut(rows_per_thread * output_dim).enumerate()
            {
                let first_row = chunk_index * rows_per_thread;
                scope.spawn(move || compute(first_row, chunk));
            }
        });
    }

    Tensor::new(vec![rows, output_dim], output)
}

pub(crate) fn parallel_threads(work: usize, rows: usize) -> usize {
    if work < PARALLEL_FLOP_THRESHOLD || rows < 2 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .min(rows)
}

pub(crate) fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let mut count = 1usize;
    for &dimension in shape {
        count = count
            .checked_mul(dimension)
            .ok_or_else(|| Error::new("Tensor element count exceeds usize range"))?;
    }
    Ok(count)
}

fn calculate_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for index in (0..shape.len()).rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(shape[index])
            .ok_or_else(|| Error::new("Tensor strides exceed usize range"))?;
    }
    Ok(strides)
}

pub fn shape_string(shape: &[usize]) -> String {
    let mut output = String::from("[");
    for (index, dimension) in shape.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        output.push_str(&dimension.to_string());
    }
    output.push(']');
    output
}
