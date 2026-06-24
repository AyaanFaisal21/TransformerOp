// Toolchain smoke test for PyTorch CUDA extensions.
//
// Proves the full path works on this machine: nvcc + MSVC compile a kernel,
// link against libtorch, and pybind exposes it to Python. Not a real kernel --
// just elementwise add. Every Phase 3 kernel reuses this exact skeleton:
//   __global__ kernel  ->  C++ launcher taking torch::Tensor  ->  PYBIND11 binding.

#include <torch/extension.h>
#include <cuda_runtime.h>

// The kernel: runs on the GPU, one thread per element.
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) out[i] = a[i] + b[i];                // guard: last block may overhang
}

// The launcher: C++ glue that PyTorch calls. Takes/returns torch::Tensor.
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "size mismatch");
    a = a.contiguous();
    b = b.contiguous();
    auto out = torch::empty_like(a);

    int n = a.numel();
    int threads = 256;                       // threads per block
    int blocks = (n + threads - 1) / threads;  // enough blocks to cover all n
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

// The binding: exposes add_cuda to Python as <module>.add(...).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "elementwise add (CUDA)");
}
