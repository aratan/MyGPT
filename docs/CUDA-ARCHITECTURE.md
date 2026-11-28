# Arquitectura CUDA de MyGPT

## Estado real

- **CPU:** backend actual y referencia numérica; los tensores y el entrenamiento siguen usando memoria host.
- **CUDA Driver Runtime y memoria:** primera capa implementada en la rama de trabajo. Carga dinámicamente `libcuda`, inicializa el driver, enumera dispositivos, consulta compute capability/VRAM/versión, administra un contexto RAII y reserva buffers FP32 residentes con transferencias H2D/D2H explícitas.
- **CUDA tensorial:** todavía no implementado.
- **Entrenamiento e inferencia GPU:** todavía no implementados. No se debe mostrar CUDA como backend operativo hasta completar y probar estas fases en hardware NVIDIA.

La compilación del feature `cuda` no necesita el CUDA Toolkit ni enlaza bibliotecas NVIDIA. En ejecución sí necesita el driver NVIDIA (`libcuda.so.1`).

## Objetivo de arquitectura

No convertir cada operación CPU en una copia H2D/D2H. La ruta CUDA debe mantener pesos, activaciones, gradientes y estado de AdamW residentes en VRAM durante una operación completa de entrenamiento o inferencia.

### Capas

1. **Runtime y dispositivo** — carga dinámica del CUDA Driver API, inventario de GPU, capacidades, memoria y ciclo de vida de contextos. Esta rama implementa el descubrimiento, contexto RAII y buffers FP32.
2. **Ejecución** — streams, kernels NVRTC/cuBLAS y sincronización. Los buffers FP32 propietarios y las transferencias explícitas ya están implementados; faltan streams y kernels.
3. **Primitivas tensoriales** — matmul, elementwise, reducciones, LayerNorm, softmax y activaciones. Pruebas diferenciales contra CPU.
4. **Modelo** — reutilizar las definiciones y formatos de parámetros actuales; introducir una ruta de ejecución backend explícita sin duplicar la arquitectura Transformer.
5. **Entrenamiento** — forward, loss, backward, clipping y AdamW en GPU. Estado del optimizador y gradientes residentes.
6. **Inferencia** — pesos residentes, transferencia de tokens/logits pequeños y ninguna transferencia del modelo completo por token.
7. **Operación** — selección `cpu|cuda|auto`, límites de VRAM, errores OOM, benchmarks reproducibles y CI con GPU.

## Decisiones que evitan deuda técnica

- Mantener `Tensor` CPU como referencia correcta y camino de compatibilidad.
- No introducir una variante GPU dentro de `Tensor` hasta que exista una API de backend que pueda ejecutar las operaciones completas sin descargar datos en cada llamada.
- Separar la memoria GPU (`CudaBuffer`) de la forma lógica del tensor (`shape/strides/dtype`).
- El backend debe poseer contexto y streams; prohibido usar un contexto global oculto.
- Las transferencias host/device serán explícitas y estarán en los límites de entrada/salida, checkpoints y métricas.
- FP32 es la referencia inicial. FP16/BF16 se añaden después de validar tolerancias y acumulación FP32.
- `--device cuda` nunca puede hacer fallback silencioso a CPU. `auto` solo elegirá CUDA cuando el backend completo esté disponible, no por el mero hecho de detectar una GPU.

## Validación de la primera capa

En CI se compila y ejecuta el conjunto de pruebas con `--all-features`. Las pruebas que requieren una GPU física se ejecutarán en un runner NVIDIA o en la máquina objetivo; la CI CPU no fingirá validar kernels CUDA.
