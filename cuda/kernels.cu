// Backend CUDA mínimo para las operaciones dominantes del entrenamiento.
//
// El cálculo se hace en la GPU con cuBLAS; los buffers de dispositivo se
// reservan una sola vez y se reutilizan entre llamadas. Los datos se copian
// desde/hacia el host en cada operación, así que la ganancia aparece cuando la
// matmul es suficientemente grande para amortizar el tráfico PCIe.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <mutex>

namespace {

std::mutex g_mutex;
cublasHandle_t g_handle = nullptr;
float* g_a = nullptr;
float* g_b = nullptr;
float* g_c = nullptr;
size_t g_a_capacity = 0;
size_t g_b_capacity = 0;
size_t g_c_capacity = 0;
bool g_initialized = false;

bool ensure_capacity(float** buffer, size_t* capacity, size_t bytes) {
    if (*capacity >= bytes) {
        return true;
    }
    if (*buffer != nullptr) {
        cudaFree(*buffer);
        *buffer = nullptr;
        *capacity = 0;
    }
    if (cudaMalloc(reinterpret_cast<void**>(buffer), bytes) != cudaSuccess) {
        return false;
    }
    *capacity = bytes;
    return true;
}

bool initialize() {
    if (g_initialized) {
        return true;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        return false;
    }
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    g_initialized = true;
    return true;
}

constexpr int kOk = 0;
constexpr int kNotInitialized = 1;
constexpr int kMemory = 2;
constexpr int kCublas = 3;
constexpr int kCopy = 4;
constexpr int kInvalidArguments = 5;

}  // namespace

extern "C" {

const char* mini_gpt_cuda_device_name() {
    static char name[256] = {0};
    if (name[0] != '\0') {
        return name;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return "GPU desconocida";
    }
    int written = 0;
    for (; written < 255 && properties.name[written] != '\0'; ++written) {
        name[written] = properties.name[written];
    }
    name[written] = '\0';
    return name;
}

// C (m x n, fila mayor) = A (m x k) * B (k x n).
// cuBLAS es column-major: interpretando A, B y C como sus transpuestas se
// obtiene C^T = B^T * A^T, es decir el producto pedido sin transponer en host.
int mini_gpt_cuda_matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return kInvalidArguments;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!initialize()) {
        return kNotInitialized;
    }
    const size_t a_bytes = sizeof(float) * static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(m) * static_cast<size_t>(n);
    if (!ensure_capacity(&g_a, &g_a_capacity, a_bytes) ||
        !ensure_capacity(&g_b, &g_b_capacity, b_bytes) ||
        !ensure_capacity(&g_c, &g_c_capacity, c_bytes)) {
        return kMemory;
    }
    if (cudaMemcpy(g_a, a, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(g_b, b, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return kCopy;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasSgemm(
        g_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        g_b,
        n,
        g_a,
        k,
        &beta,
        g_c,
        n);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return kCublas;
    }
    if (cudaMemcpy(c, g_c, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return kCopy;
    }
    return kOk;
}

}  // extern "C"
