# Validación CUDA en NVIDIA

Este documento separa lo que demuestra GitHub Actions de lo que debe probarse en una GPU real.

## 1. Qué valida cada entorno

| Entorno | Qué demuestra | Qué no demuestra |
|---|---|---|
| GitHub Actions `ubuntu-latest` | Formato, Clippy, tests host y compilación con `--all-features` | Ejecución de kernels en una GPU NVIDIA |
| Linux con driver NVIDIA | Descubrimiento del dispositivo, carga de Driver API/NVRTC/cuBLAS y tests CUDA disponibles | Rendimiento representativo si no es la RTX objetivo |
| RTX 4060 8 GB | Paridad numérica, estabilidad y rendimiento del hardware objetivo | Generalización a otras GPU |

Los tests que necesitan CUDA detectan si el runtime/dispositivo está disponible. Que terminen sin error en un runner sin GPU no debe interpretarse como una prueba física del backend.

## 2. Requisitos en Arch Linux

- Driver NVIDIA instalado y operativo.
- GPU visible mediante `nvidia-smi`.
- Bibliotecas del driver, NVRTC y cuBLAS accesibles por el cargador dinámico.
- Rust estable y Cargo.

El backend carga dinámicamente CUDA Driver API, NVRTC y cuBLAS. No necesita `nvcc`, CUDA Toolkit completo, CMake ni `build.rs` para compilar el proyecto; sí necesita las bibliotecas de runtime correspondientes al ejecutar.

Comprobaciones iniciales:

```bash
nvidia-smi
rustc --version
cargo --version
```

## 3. Compilar y ejecutar pruebas

Desde la raíz del repositorio:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked
cargo test --all-targets --all-features --locked
cargo build --release --all-features --locked
```

Para registrar el resultado:

```bash
nvidia-smi
cargo test --all-targets --all-features --locked -- --nocapture
```

Si una prueba CUDA indica que no hay dispositivo o se omite, revisar primero el driver y las bibliotecas dinámicas. No atribuir ese resultado a una validación de kernels.

## 4. Estado funcional actual

- `CudaTransformerBlock`: importa parámetros de un bloque CPU y ejecuta LayerNorm, atención causal multi-cabeza, residuales y MLP/GELU en CUDA.
- `CudaGPT::from_cpu`: copia embeddings, bloques, normalización final y proyección de salida a VRAM.
- `CudaGPT::forward`: devuelve logits del GPT completo en un tensor CUDA.
- PR #17 añade `CudaGPT::generate`: reutiliza la configuración de sampling del proyecto, pero descarga logits a CPU en cada paso y vuelve a calcular el contexto completo.

La generación actual todavía no dispone de KV cache ni sampling residente en GPU. Tampoco se ha confirmado aquí una ejecución física en RTX 4060.

## 5. Datos que deben registrarse en la prueba RTX 4060

Anotar junto al resultado:

- Modelo exacto de GPU y versión del driver.
- Versiones de Rust y sistema operativo.
- Resultado de cada comando anterior.
- Resultado de tests de paridad CPU/CUDA.
- Longitud de prompt, tokens generados, latencia por token y pico de VRAM.
- Cualquier error de NVRTC, cuBLAS o Driver API.

No publicar un benchmark como rendimiento CUDA validado si la ejecución se hizo en CPU o si los tests de GPU fueron omitidos.
