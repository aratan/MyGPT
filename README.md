# mini_gpt (Rust)

Port a Rust del proyecto C++ `mini-gpt-cpp`. Incluye el núcleo completo:

- `tensor`: tensor denso 2D/3D en fila mayor, operaciones elemento a elemento,
  `matmul` y `transpose`.
- `tokenizer`: Byte-level BPE con los mismos tokens especiales y formato
  binario `MINIGPTB` que el original.
- `embedding`, `layer_norm`, `attention`, `transformer`, `gpt`.
- `loss`, `backprop`, `adamw`: entrenamiento completo (cross-entropy,
  retropropagación de todas las capas, AdamW y recorte de norma global).
- `data`, `training_state`: corpus UTF-8, separación train/validación,
  batches con máscara y reanudación exacta del muestreo.
- `inference`, `cli`.

## Compilar y probar

```bash
cd rust
cargo build --release            # CPU
cargo build --release --features cuda   # GPU NVIDIA con CUDA Toolkit
cargo test
cargo test --features cuda
```

La guía completa (diagramas, uso de la CLI, CUDA en la RTX 4060 y medidas de
rendimiento) está en el [README principal](../README.md).

Binarios resultantes:

- `target/release/tokenizer_train`
- `target/release/gpt_train`
- `target/release/gpt_chat`

Son compatibles en argumentos con los del proyecto C++.

## Compatibilidad de checkpoints

Los formatos binarios (`MINIEMBD`, `MINIATTN`, `MINITRFB`, `MINIGPTM`,
`MINIADAM`) y el estado de entrenamiento (`*.data`) se conservan byte a byte,
incluido el orden de los tensores. Se verificó la carga cruzada en ambos
sentidos: un modelo guardado por C++ se carga y ejecuta en Rust, y un modelo
guardado por Rust se carga y ejecuta en C++.

La inicialización aleatoria usa un generador distinto (xoshiro256** frente a
`std::mt19937`), así que dos modelos nuevos creados con la misma semilla no
tienen los mismos pesos. Los checkpoints ya guardados sí son intercambiables.

## Optimizaciones respecto al original

- Matmul repartida por filas entre hilos (`std::thread::scope`) por encima de
  un umbral de trabajo, con bucles `i-k-j` sobre `&[f32]` que evitan índices
  por elemento y permiten vectorizar el bucle interno.
- Proyecciones lineales 3D ejecutadas sobre el buffer plano del lote, sin
  materializar tensores 2D intermedios.
- Entrenamiento BPE con conteo incremental de pares (en lugar de reconstruir
  el mapa completo en cada merge) y un hasher rápido (tipo FxHash).
- `reshape` sin copia, movimiento de tensores en lugar de clones en el
  backward, y perfil `release` con LTO, `codegen-units = 1` y `opt-level = 3`.
- Sin dependencias externas.

No se ha medido una comparación de rendimiento frente a la versión C++; las
comprobaciones realizadas son los tests y el flujo completo de extremo a
extremo.

## Limitaciones

- No se porta la ruta CUDA (`src/cuda`); este port usa CPU.
- No se portaron los binarios de demostración (`apps/*_demo.cpp`), solo las
  tres herramientas de producción y la librería.
- `gpt_train` no captura Ctrl+C: hay que esperar al final del paso o del
  entrenamiento. Los checkpoints periódicos (`--save-every`) siguen activos.
- La GELU usa una aproximación de `erf` (Abramowitz–Stegun 7.1.26) porque
  `f64::erf` no está en la librería estándar de Rust; el error es del orden
  de 1e-7, despreciable para entrenamiento.
