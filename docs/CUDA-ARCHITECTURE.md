# Arquitectura CUDA

## Estado

La integración CUDA es optativa mediante `--features cuda` y actualmente solo se habilita en Linux. El build por defecto continúa siendo CPU-only y no requiere CUDA Toolkit.

| Capa | Estado |
|---|---|
| CUDA Driver API dinámica | Implementada |
| Descubrimiento de dispositivos, contexto, buffers FP32 y streams | Implementada |
| cuBLAS SGEMM FP32 | Implementada |
| Asociación cuBLAS a streams explícitos | Implementada |
| NVRTC: CUDA C++ → PTX en ejecución | Implementada en código; requiere runtime NVIDIA |
| Carga de módulos y lanzamiento de kernels | Implementada en código |
| Elementwise add/mul y escala FP32 | Implementadas en código |
| Reducción FP32 a escalar | Implementada en código |
| Softmax estable por fila | Implementada en código |
| LayerNorm por fila (varianza en dos pasadas) | Implementada en código |
| Validación numérica real en RTX 4060 | Pendiente |
| Integración con Tensor/Transformer | Pendiente |
| Backward, clipping y AdamW CUDA | Pendiente |
| Entrenamiento e inferencia GPU | Pendientes |

## Diseño

- `CudaRuntime` carga el driver NVIDIA dinámicamente y enumera dispositivos.
- `CudaContext` posee un contexto RAII.
- `CudaBuffer` mantiene los datos FP32 en VRAM. Las transferencias host/device son explícitas.
- `CudaStream` controla la cola de trabajo.
- `CudaBlas` ejecuta SGEMM sobre buffers residentes; el handle puede asociarse a un stream explícito.
- `CudaKernelModule` compila fuentes CUDA con NVRTC, carga el PTX y expone funciones.
- `CudaKernelModule::compile_builtin_ops` compila los kernels iniciales. Sus operaciones encolan trabajo sin copiar datos a RAM ni sincronizar automáticamente.

## Arquitectura objetivo

La RTX 4060 usa compute capability 8.9, por tanto NVRTC recibe `compute_89`. FP32 es la referencia inicial. FP16/BF16 se evaluarán después, con acumulación FP32 y pruebas de tolerancia.

## Validación requerida

1. Ejecutar en GPU real la compilación NVRTC y cada kernel.
2. Comparar resultados con las operaciones CPU usando entradas aleatorias, casos límite y tolerancias explícitas.
3. Medir errores numéricos de LayerNorm/softmax y tamaños no múltiplos de 256.
4. Integrar un tensor CUDA residente sin transferencias por operación.
5. Conectar forward, backward y AdamW antes de declarar entrenamiento/inferencia CUDA disponibles.

La CI CPU con `--all-features` solo verifica compilación y tests que no necesitan GPU. No demuestra que NVRTC o los kernels se hayan ejecutado físicamente.
