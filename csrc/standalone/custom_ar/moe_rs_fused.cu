// Lamport reduce-scatter fused with residual add + RMSNorm.
//
// Replaces three separate kernels (RS + residual_add + RMSNorm) with one:
//   1. PUSH: write MoE output to all peers' Lamport buffers.
//   2. CLEAR: write sentinels to old segment.
//   3. POLL+REDUCE+FUSE (per-token):
//      a. Volatile-load from all peers, sum in fp32.
//      b. Add residual.
//      c. Compute RMSNorm (block reduction for variance).
//      d. Store normed output + updated residual.
//
// Saves: one kernel launch (~3-5µs) + one global memory round-trip per layer.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define DINLINE __device__ __forceinline__

constexpr uint32_t SENTINEL = 0x80000000u;
constexpr int kMaxBlocks = 36;

// Each token has D=7168 bf16 values = 896 int4 vectors.
// With 512 threads: ceil(896/512) = 2 int4 per thread = 16 fp32 values.
constexpr int kMaxValsPerThread = 16;

static DINLINE int4 ld128v(const void* addr) {
  int4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr));
  return v;
}

static DINLINE bool has_sentinel(int4 v) {
  return reinterpret_cast<uint32_t&>(v.x) == SENTINEL |
         reinterpret_cast<uint32_t&>(v.y) == SENTINEL |
         reinterpret_cast<uint32_t&>(v.z) == SENTINEL |
         reinterpret_cast<uint32_t&>(v.w) == SENTINEL;
}

static DINLINE int4 remove_sentinel(int4 v) {
  if (reinterpret_cast<uint32_t&>(v.x) == SENTINEL) v.x = 0;
  if (reinterpret_cast<uint32_t&>(v.y) == SENTINEL) v.y = 0;
  if (reinterpret_cast<uint32_t&>(v.z) == SENTINEL) v.z = 0;
  if (reinterpret_cast<uint32_t&>(v.w) == SENTINEL) v.w = 0;
  return v;
}

// bf16 helpers
static DINLINE void accumulate_bf16(float* acc, int4 v) {
  const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
  for (int k = 0; k < 8; k++) acc[k] += __bfloat162float(bp[k]);
}

static DINLINE void add_bf16_to_fp32(float* dst, int4 v) {
  const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
  for (int k = 0; k < 8; k++) dst[k] += __bfloat162float(bp[k]);
}

static DINLINE int4 fp32_to_bf16_int4(const float* vals) {
  int4 out;
  __nv_bfloat16* bp = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int k = 0; k < 8; k++) bp[k] = __float2bfloat16(vals[k]);
  return out;
}

// Block-level tree reduction in shared memory.
static DINLINE float block_reduce_sum(float val, float* smem) {
  smem[threadIdx.x] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  return smem[0];
}

// ---------------------------------------------------------------------------
// Fused reduce-scatter + residual + RMSNorm kernel
// ---------------------------------------------------------------------------

template <int ngpus>
__global__ void __launch_bounds__(512, 1) moe_rs_fused_kernel(
    int64_t* buf_ptrs, int* counters, int rank, int seg_capacity,
    int rank_stride,
    const void* input,        // [N_total, D] bf16 — MoE output
    const void* residual_in,  // [N_per_rank, D] bf16 — skip connection
    const void* gamma,        // [D] bf16 — RMSNorm weight
    void* normed_out,         // [N_per_rank, D] bf16 — normed result
    void* residual_out,       // [N_per_rank, D] bf16 — updated residual
    int total_sz,             // int4 units of full input
    int slice_off,            // int4 offset to this rank's slice
    int slice_sz,             // int4 units of this rank's slice
    int D_int4,               // int4 units per token (hidden_dim * 2 / 16)
    int N_per_rank,           // tokens in this rank's slice
    float eps) {              // RMSNorm epsilon
  using V = int4;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  const int seg = counters[1];
  const int prev_total_sz = counters[2];
  const int cur_seg = seg;
  const int old_seg = 1 - seg;

  char* bufs[ngpus];
#pragma unroll
  for (int r = 0; r < ngpus; r++)
    bufs[r] = reinterpret_cast<char*>(buf_ptrs[r]) + cur_seg * seg_capacity;

  V sent;
  sent.x = sent.y = sent.z = sent.w = static_cast<int>(SENTINEL);

  // ---- Phase 1: PUSH full MoE output to ALL peers ----
  {
    const V* src = reinterpret_cast<const V*>(input);
    for (int i = tid; i < total_sz; i += stride) {
      V val = remove_sentinel(src[i]);
#pragma unroll
      for (int r = 0; r < ngpus; r++)
        reinterpret_cast<V*>(bufs[r] + rank * rank_stride)[i] = val;
    }
  }

  // ---- Phase 2: CLEAR old segment ----
  if (prev_total_sz > 0) {
    char* clr_base =
        reinterpret_cast<char*>(buf_ptrs[rank]) + old_seg * seg_capacity;
#pragma unroll
    for (int r = 0; r < ngpus; r++) {
      V* clr = reinterpret_cast<V*>(clr_base + r * rank_stride);
      for (int i = tid; i < prev_total_sz; i += stride) clr[i] = sent;
    }
  }

  // ---- Phase 3: Fused POLL + REDUCE + RESIDUAL + RMSNORM ----
  // Each block handles one token. Only first N_per_rank blocks participate.
  if (blockIdx.x < N_per_rank) {
    const int token = blockIdx.x;
    const int token_off = slice_off + token * D_int4;  // in the full buffer
    char* my = bufs[rank];

    extern __shared__ float smem[];

    // Register storage for intermediate fp32 values.
    float local_vals[kMaxValsPerThread];
    int n_vals = 0;
    float partial_sum_sq = 0.0f;

    // Pass 1: poll + reduce + add residual + compute sum_sq
    for (int pos = threadIdx.x; pos < D_int4; pos += blockDim.x) {
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

      // Poll all ranks' data for this position.
#pragma unroll
      for (int s = 0; s < ngpus; s++) {
        V val;
        do {
          val = ld128v(reinterpret_cast<V*>(my + s * rank_stride) + token_off +
                       pos);
        } while (has_sentinel(val));
        accumulate_bf16(acc, val);
      }

      // Add residual.
      V res = reinterpret_cast<const V*>(residual_in)[token * D_int4 + pos];
      add_bf16_to_fp32(acc, res);

      // Store in registers and accumulate sum_sq.
#pragma unroll
      for (int k = 0; k < 8; k++) {
        local_vals[n_vals++] = acc[k];
        partial_sum_sq += acc[k] * acc[k];
      }
    }

    // Block-level reduction: total sum of squares.
    float total_sum_sq = block_reduce_sum(partial_sum_sq, smem);
    float rms_scale = rsqrtf(total_sum_sq / (D_int4 * 8) + eps);

    // Pass 2: apply RMSNorm, store outputs.
    n_vals = 0;
    const V* gamma_v = reinterpret_cast<const V*>(gamma);
    for (int pos = threadIdx.x; pos < D_int4; pos += blockDim.x) {
      V gv = gamma_v[pos];
      const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);

      // Build normed output and residual output.
      float normed_fp32[8], res_fp32[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float val = local_vals[n_vals++];
        res_fp32[k] = val;  // residual_out
        normed_fp32[k] = val * rms_scale * __bfloat162float(gp[k]);  // normed
      }

      reinterpret_cast<V*>(normed_out)[token * D_int4 + pos] =
          fp32_to_bf16_int4(normed_fp32);
      reinterpret_cast<V*>(residual_out)[token * D_int4 + pos] =
          fp32_to_bf16_int4(res_fp32);
    }
  }

  // ---- Phase 4: ADVANCE ----
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counters[1] = 1 - seg;
    counters[2] = total_sz;
  }
}

// ---------------------------------------------------------------------------
// Sentinel init + host launcher
// ---------------------------------------------------------------------------

__global__ void lamport_init_kernel(uint32_t* buf, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) buf[i] = SENTINEL;
}

void lamport_init(int64_t buf_ptr, int64_t nbytes) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  int n = static_cast<int>(nbytes / 4);
  lamport_init_kernel<<<256, 256, 0, stream>>>(
      reinterpret_cast<uint32_t*>(buf_ptr), n);
}

void moe_rs_fused(int64_t buf_ptrs_ptr, int64_t counters_ptr, int64_t rank,
                  int64_t world_size, int64_t seg_capacity, int64_t rank_stride,
                  torch::Tensor input,         // [N_total, D] bf16
                  torch::Tensor residual_in,   // [N_per_rank, D] bf16
                  torch::Tensor gamma,         // [D] bf16
                  torch::Tensor normed_out,    // [N_per_rank, D] bf16
                  torch::Tensor residual_out,  // [N_per_rank, D] bf16
                  double eps) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(input.is_contiguous() && residual_in.is_contiguous());
  TORCH_CHECK(gamma.is_contiguous() && normed_out.is_contiguous());
  TORCH_CHECK(residual_out.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);

  int ws = static_cast<int>(world_size);
  int r = static_cast<int>(rank);
  int64_t N_total = input.size(0);
  int64_t D = input.size(1);
  TORCH_CHECK(N_total % ws == 0);
  int N_per_rank = static_cast<int>(N_total / ws);

  int64_t input_bytes = input.numel() * input.element_size();
  TORCH_CHECK(input_bytes % 16 == 0);
  TORCH_CHECK(input_bytes <= rank_stride);

  int total_sz = static_cast<int>(input_bytes / 16);
  int D_int4 = static_cast<int>(D * 2 / 16);  // bf16 elements → int4 units
  int slice_sz = total_sz / ws;
  int slice_off = r * slice_sz;

  int threads = 512;
  // Need at least N_per_rank blocks for Phase 3 (one per token).
  int blocks = std::max(
      N_per_rank, std::min(kMaxBlocks, (total_sz + threads - 1) / threads));

  auto* bp = reinterpret_cast<int64_t*>(buf_ptrs_ptr);
  auto* ct = reinterpret_cast<int*>(counters_ptr);
  int sc = static_cast<int>(seg_capacity);
  int rs = static_cast<int>(rank_stride);
  int smem = threads * sizeof(float);

#define LAUNCH(ng)                                                      \
  moe_rs_fused_kernel<ng><<<blocks, threads, smem, stream>>>(           \
      bp, ct, r, sc, rs, input.data_ptr(), residual_in.data_ptr(),      \
      gamma.data_ptr(), normed_out.data_ptr(), residual_out.data_ptr(), \
      total_sz, slice_off, slice_sz, D_int4, N_per_rank,                \
      static_cast<float>(eps));

  switch (ws) {
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 6:
      LAUNCH(6);
      break;
    case 8:
      LAUNCH(8);
      break;
    default:
      TORCH_CHECK(false, "world_size must be 2, 4, 6, or 8");
  }
#undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_rs_fused", &moe_rs_fused,
        "Fused Lamport reduce-scatter + residual + RMSNorm");
  m.def("lamport_init", &lamport_init,
        "Initialize Lamport buffer with sentinels");
}
