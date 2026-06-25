// Fused causal self-attention (mini-FlashAttention).
//
// Online-softmax recurrence; the (T,T) score matrix is never materialized.
// Progression (numbers in benchmarks/kernels.md):
//   v1: thread/row. Correct; acc[64] spills to local mem. ~100% occupancy. 7.78ms
//   v2: warp/row, dims split across lanes -> a warp reduction PER KEY. Regressed.
//   v3: warp/row, lane=key (efficient inner loop) but ONE WARP PER BLOCK -> ~6%
//       occupancy (16KB shared/block starves the SM). Regressed hard (30ms).
//   v4: QPB queries per block sharing ONE K/V tile. Restored occupancy (11ms) but
//       the dot read Ktile[lane*HS+k] -> all 32 lanes hit one shared-memory bank
//       (32-way conflict, serialized).
//   v5 (ACTIVE): v4 + K stored TRANSPOSED in shared memory (Ktile[k*TILE+key]) so
//       the dot's 32 lanes hit 32 distinct banks -> conflict-free reads.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define HS    64
#define WARP  32
#define TILE  32     // keys per shared tile
#define QPB   8      // queries (= warps) per block; shares the K/V tile

__device__ __forceinline__ float warpMax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float warpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__global__ void attn_kernel_v5(const float* Q, const float* K, const float* V,
                               float* O, int B_nh, int T, float scale) {
    int tid = threadIdx.x;
    int w = tid / WARP;              // which query (warp) in this block
    int lane = tid % WARP;
    long long base_row = (long long)blockIdx.x * QPB;
    long long row = base_row + w;    // this warp's query row
    int seq = row / T;
    int i = row % T;                 // this warp's causal limit

    __shared__ float Ktile[TILE * HS];   // TRANSPOSED: [k][key] -> conflict-free dot reads
    __shared__ float Vtile[TILE * HS];   // natural: [key][d]
    __shared__ float qs[QPB * HS];       // each warp's query
    __shared__ float ps[QPB * WARP];     // each warp's per-key weights

    const float* Kseq = K + (long long)seq * T * HS;
    const float* Vseq = V + (long long)seq * T * HS;

    qs[w * HS + lane] = Q[row * HS + lane];
    qs[w * HS + lane + WARP] = Q[row * HS + lane + WARP];
    __syncthreads();

    float m = -FLT_MAX, l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
    int max_i = (int)((base_row + QPB - 1) % T);   // block-wide last position

    for (int tile_start = 0; tile_start <= max_i; tile_start += TILE) {
        // all 256 threads load the shared K/V tile once (reused by all QPB queries)
        for (int idx = tid; idx < TILE * HS; idx += blockDim.x) {
            // V natural layout [key][d]
            int vgj = tile_start + idx / HS;
            Vtile[idx] = (vgj < T) ? Vseq[(long long)vgj * HS + idx % HS] : 0.0f;
            // K transposed layout [k][key]: Ktile[k*TILE + key] = K[key, k]
            int kgj = tile_start + idx % TILE;
            Ktile[idx] = (kgj < T) ? Kseq[(long long)kgj * HS + idx / TILE] : 0.0f;
        }
        __syncthreads();

        if (tile_start <= i) {                 // warp-uniform: this query has keys here
            int gj = tile_start + lane;
            float s = -FLT_MAX;
            if (gj <= i) {
                float d = 0.0f;
                #pragma unroll
                for (int k = 0; k < HS; k++) d += qs[w * HS + k] * Ktile[k * TILE + lane];
                s = d * scale;
            }
            float new_m = fmaxf(m, warpMax(s));
            float corr = expf(m - new_m);
            float p = (s == -FLT_MAX) ? 0.0f : expf(s - new_m);
            ps[w * WARP + lane] = p;
            l = l * corr + warpSum(p);
            __syncwarp();

            float u0 = 0.0f, u1 = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < TILE; kk++) {
                float pk = ps[w * WARP + kk];
                u0 += pk * Vtile[kk * HS + lane];
                u1 += pk * Vtile[kk * HS + lane + WARP];
            }
            acc0 = acc0 * corr + u0;
            acc1 = acc1 * corr + u1;
            m = new_m;
        }
        __syncthreads();                       // before the next tile overwrites shared K/V
    }

    float inv = 1.0f / l;
    O[row * HS + lane] = acc0 * inv;
    O[row * HS + lane + WARP] = acc1 * inv;
}

torch::Tensor attn_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "float32 only");
    TORCH_CHECK(Q.size(-1) == HS, "this kernel is compiled for head size 64");
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    int B = Qc.size(0), nh = Qc.size(1), T = Qc.size(2);
    int B_nh = B * nh;
    TORCH_CHECK(((long long)B_nh * T) % QPB == 0 && T % QPB == 0,
                "B*nh*T and T must be divisible by QPB");
    auto O = torch::empty_like(Qc);

    float scale = 1.0f / sqrtf((float)HS);
    long long blocks = (long long)B_nh * T / QPB;
    attn_kernel_v5<<<blocks, QPB * WARP>>>(
        Qc.data_ptr<float>(), Kc.data_ptr<float>(), Vc.data_ptr<float>(),
        O.data_ptr<float>(), B_nh, T, scale);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn", &attn_cuda, "fused causal self-attention (CUDA)");
}
