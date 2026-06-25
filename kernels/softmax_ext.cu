// Custom row-wise softmax kernel (first real Phase 3 kernel).
//
// Two designs, dispatched by shape at runtime (like torch does):
//   - warp kernel: one WARP (32 threads) per row, warp-shuffle reductions. Wins
//     on SHORT rows (few cols -> 32 lanes is enough; no shared mem / no barriers).
//   - block kernel: one BLOCK (256 threads) per row, shared-memory tree reduction.
//     Wins on WIDE rows (many cols -> needs >32 threads/row to parallelize).
// Measured crossover sits between 256 and 1024 cols; threshold set accordingly.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// --- warp all-reduce (butterfly): after 5 shuffles every lane holds the result ---
__device__ __forceinline__ float warpReduceMax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}
__device__ __forceinline__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// one warp per row -- best for short rows
__global__ void softmax_warp_kernel(const float* x, float* out, int rows, int cols) {
    int warps_per_block = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * warps_per_block + warp_in_block;
    if (row >= rows) return;
    const float* xr = x + (long long)row * cols;
    float* outr = out + (long long)row * cols;

    float m = -FLT_MAX;
    for (int i = lane; i < cols; i += 32) m = fmaxf(m, xr[i]);
    m = warpReduceMax(m);

    float s = 0.0f;
    for (int i = lane; i < cols; i += 32) s += expf(xr[i] - m);
    s = warpReduceSum(s);

    float inv = 1.0f / s;
    for (int i = lane; i < cols; i += 32) outr[i] = expf(xr[i] - m) * inv;
}

// one block per row, shared-memory tree reduction -- best for wide rows
__global__ void softmax_block_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (long long)row * cols;
    float* outr = out + (long long)row * cols;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += nthreads) local_max = fmaxf(local_max, xr[i]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += nthreads) local_sum += expf(xr[i] - row_max);
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    float inv = 1.0f / row_sum;
    for (int i = tid; i < cols; i += nthreads) outr[i] = expf(xr[i] - row_max) * inv;
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "float32 only");
    auto xc = x.contiguous();
    auto shape = xc.sizes().vec();
    int cols = shape.back();
    int rows = xc.numel() / cols;
    auto out = torch::empty_like(xc);

    const int threads = 256;
    const int WARP_ROW_MAX_COLS = 512;  // heuristic crossover (warp wins below, block above)

    if (cols <= WARP_ROW_MAX_COLS) {
        int warps_per_block = threads / 32;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;
        softmax_warp_kernel<<<blocks, threads>>>(
            xc.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
    } else {
        size_t shmem = threads * sizeof(float);
        softmax_block_kernel<<<rows, threads, shmem>>>(
            xc.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
    }
    return out.view(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_cuda, "row-wise softmax over the last dim (CUDA, shape-dispatched)");
}
