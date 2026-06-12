// Toolchain smoke test: verifies nvcc + MSVC can build for sm_75 and the
// driver can run the result. Not part of the project proper.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    float *a, *b, *c;
    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    add<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    for (int i = 0; i < n; i++) {
        if (c[i] != 3.0f) { printf("FAIL at %d: %f\n", i, c[i]); return 1; }
    }
    printf("smoke test OK: %d elements, kernel ran on device\n", n);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
