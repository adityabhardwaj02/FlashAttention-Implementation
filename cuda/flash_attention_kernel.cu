// Part D: FlashAttention forward pass, custom CUDA kernel.
// Grid: (B, NH) -- one block per (batch, head) slice. Block: 32 threads.
// Outer loop over K/V tiles, inner loop over Q tiles (per assignment spec).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#define BR 32
#define BC 32

extern __shared__ float smem[];

__global__ void flash_attention_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ m_buf,
    float* __restrict__ l_buf,
    float* __restrict__ O_acc,
    int N, int d, int NH, float scale)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tx = threadIdx.x;

    const int Tc = (N + BC - 1) / BC;
    const int Tr = (N + BR - 1) / BR;

    const long qkv_base = (long)(b * NH + h) * N * d;
    const long lm_base  = (long)(b * NH + h) * N;

    // Shared tiles: Qi (Br x d), Kj (Bc x d), Vj (Bc x d), S (Br x Bc)
    float* Qi = smem;
    float* Kj = Qi + BR * d;
    float* Vj = Kj + BC * d;
    float* S  = Vj + BC * d;

    for (int j = 0; j < Tc; j++) {
        int kv_row = j * BC + tx;
        if (kv_row < N) {
            for (int x = 0; x < d; x++) {
                long idx = qkv_base + (long)kv_row * d + x;
                Kj[tx * d + x] = __half2float(K[idx]);
                Vj[tx * d + x] = __half2float(V[idx]);
            }
        }
        __syncthreads();

        int bc_valid = min(BC, N - j * BC);

        for (int i = 0; i < Tr; i++) {
            int q_row = i * BR + tx;
            if (q_row >= N) continue;

            for (int x = 0; x < d; x++) {
                Qi[tx * d + x] = __half2float(Q[qkv_base + (long)q_row * d + x]);
            }

            float m_prev = m_buf[lm_base + q_row];
            float l_prev = l_buf[lm_base + q_row];

            float row_max = -INFINITY;
            for (int y = 0; y < bc_valid; y++) {
                float acc = 0.f;
                for (int x = 0; x < d; x++) {
                    acc += Qi[tx * d + x] * Kj[y * d + x];
                }
                acc *= scale;
                S[tx * BC + y] = acc;
                row_max = fmaxf(row_max, acc);
            }

            float l_tilde = 0.f;
            for (int y = 0; y < bc_valid; y++) {
                float p = __expf(S[tx * BC + y] - row_max);
                S[tx * BC + y] = p;
                l_tilde += p;
            }

            // Online softmax merge + output rescale
            float m_new = fmaxf(m_prev, row_max);
            float corr_prev = __expf(m_prev - m_new);
            float corr_tilde = __expf(row_max - m_new);
            float l_new = corr_prev * l_prev + corr_tilde * l_tilde;

            long o_row_base = qkv_base + (long)q_row * d;
            for (int x = 0; x < d; x++) {
                float pv = 0.f;
                for (int y = 0; y < bc_valid; y++) {
                    pv += S[tx * BC + y] * Vj[y * d + x];
                }
                float o_prev = O_acc[o_row_base + x];
                O_acc[o_row_base + x] = o_prev * corr_prev + corr_tilde * pv;
            }

            m_buf[lm_base + q_row] = m_new;
            l_buf[lm_base + q_row] = l_new;
        }
        __syncthreads();
    }

    // Final normalization pass
    for (int i = 0; i < Tr; i++) {
        int q_row = i * BR + tx;
        if (q_row >= N) continue;
        float l_final = l_buf[lm_base + q_row];
        long o_row_base = qkv_base + (long)q_row * d;
        for (int x = 0; x < d; x++) {
            O[o_row_base + x] = __float2half(O_acc[o_row_base + x] / l_final);
        }
    }
}

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");

    const int B  = Q.size(0);
    const int NH = Q.size(1);
    const int N  = Q.size(2);
    const int d  = Q.size(3);
    const float scale = 1.0f / sqrtf((float)d);

    auto O = torch::empty_like(Q);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto m_buf = torch::full({B, NH, N}, -std::numeric_limits<float>::infinity(), opts_f32);
    auto l_buf = torch::zeros({B, NH, N}, opts_f32);
    auto O_acc = torch::zeros({B, NH, N, d}, opts_f32);

    dim3 grid(B, NH);
    dim3 block(BR);
    size_t smem_bytes = (size_t)(BR * d + 2 * BC * d + BR * BC) * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(flash_attention_fwd_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              smem_bytes);
    }

    flash_attention_fwd_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        m_buf.data_ptr<float>(),
        l_buf.data_ptr<float>(),
        O_acc.data_ptr<float>(),
        N, d, NH, scale);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "FlashAttention forward (CUDA)");
}
