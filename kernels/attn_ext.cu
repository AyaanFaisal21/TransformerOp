// Fused causal self-attention (mini-FlashAttention), v1: correct, not yet tuned.
//
// Computes softmax(causal(Q Kᵀ / sqrt(hs))) V WITHOUT materializing the (T, T)
// score matrix in global memory -- the whole point of fusion. It does this with
// the FlashAttention "online softmax" recurrence: stream over keys keeping a
// running max (m), running normalizer (l), and running output accumulator (acc),
// rescaling acc/l whenever a bigger score appears.
//
// v1 layout: ONE THREAD per query row. Simplest correct expression of the
// algorithm -- no shared memory, no tiling. Expected to be slower than SDPA;
// this version exists to lock in correctness before optimizing.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define HS 64   // head size (this model). asserted in the launcher.

__global__ void attn_kernel(const float* Q, const float* K, const float* V,
                            float* O, int B_nh, int T, float scale) {
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= (long long)B_nh * T) return;
    int seq = r / T;            // which (batch, head) sequence
    int i = r % T;              // this thread's query position
    const float* q = Q + r * HS;
    const float* Kseq = K + (long long)seq * T * HS;
    const float* Vseq = V + (long long)seq * T * HS;
    float* o = O + r * HS;

    float qreg[HS];
    #pragma unroll
    for (int d = 0; d < HS; d++) qreg[d] = q[d];

    float m = -FLT_MAX, l = 0.0f, acc[HS];
    #pragma unroll
    for (int d = 0; d < HS; d++) acc[d] = 0.0f;

    // stream over keys 0..i (causal: never look past the query position)
    for (int j = 0; j <= i; j++) {
        const float* kj = Kseq + (long long)j * HS;
        float s = 0.0f;
        #pragma unroll
        for (int d = 0; d < HS; d++) s += qreg[d] * kj[d];
        s *= scale;

        // online-softmax update
        float new_m = fmaxf(m, s);
        float corr = expf(m - new_m);     // rescale old running sums to the new max
        float p = expf(s - new_m);        // weight of this key
        l = l * corr + p;
        const float* vj = Vseq + (long long)j * HS;
        #pragma unroll
        for (int d = 0; d < HS; d++) acc[d] = acc[d] * corr + p * vj[d];
        m = new_m;
    }

    float inv = 1.0f / l;
    #pragma unroll
    for (int d = 0; d < HS; d++) o[d] = acc[d] * inv;
}

torch::Tensor attn_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "float32 only");
    TORCH_CHECK(Q.size(-1) == HS, "this kernel is compiled for head size 64");
    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    int B = Qc.size(0), nh = Qc.size(1), T = Qc.size(2);
    int B_nh = B * nh;
    auto O = torch::empty_like(Qc);

    float scale = 1.0f / sqrtf((float)HS);
    long long total = (long long)B_nh * T;       // one thread per query row
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    attn_kernel<<<blocks, threads>>>(
        Qc.data_ptr<float>(), Kc.data_ptr<float>(), Vc.data_ptr<float>(),
        O.data_ptr<float>(), B_nh, T, scale);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn", &attn_cuda, "fused causal self-attention (CUDA)");
}
