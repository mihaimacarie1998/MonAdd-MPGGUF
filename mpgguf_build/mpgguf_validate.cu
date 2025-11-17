// mpgguf_validate.cu  — MPGGUF v2 validator with hardened GGUF/MP readers
//
// Build:
//   nvcc -O3 -std=c++17 -o mpgguf_validate mpgguf_validate.cu
//
// Usage:
//   ./mpgguf_validate --mpgguf model.mpgguf --fp16 baseline_f16_or_f32.gguf [--report] [--diff-high-low]
//
// What it does:
//   * Loads MPGGUF v2 container (LOW/HIGH/FP slots, adjacent packing).
//   * Loads baseline GGUF (FP16 or FP32 weights) using a tolerant KV skipper and safe strings.
//   * Dequantizes HIGH (Q8_0) and legacy scalar 2-bit LOW when recognizable by size.
//   * Computes MSE, RMSE, and max|Δ| vs FP16 baseline (and optional HIGH vs LOW).
//   * Skips unknown 2-bit layouts (e.g., Q2_K / IQ2_XXS) with a clear notice.
//
// Notes:
//   * No inference — this only validates dequantized weights vs a baseline.
//   * All readers are fast-fail hardened to avoid endless parsing on malformed files.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "gguf_parser.h"
#include "mpgguf_parser.h"
#include "parser_util.h"

using namespace monadd;

// ------------- CUDA helpers -------------
#define CUDA_OK(x)                                                                                          \
    do {                                                                                                    \
        auto _e = (x);                                                                                      \
        if (_e != cudaSuccess) {                                                                            \
            std::cerr << "CUDA " << cudaGetErrorString(_e) << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

// =============== CUDA Dequant Kernels ===============
// Q8_0: per 32 values: [float scale][int8 x 32]
__global__ void k_deq_q8(const int8_t* q, const uint16_t* s, __half* out, int blocks) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= blocks) {
        return;
    }

    __half         h = __ushort_as_half(s[id]);
    float          d = __half2float(h);

    if (!isfinite(d))
        d = 0.0f;

    const int8_t* src = q + id * QK;
    __half* dst = out + id * QK;
#pragma unroll
    for (int i = 0; i < QK; i++) {
        dst[i] = __float2half(d * float(src[i]));
    }
}

namespace q2k_detail {
    static constexpr int    QK_K = 256;
    static constexpr int    SCALES_SZ = QK_K / 16; // 16
    static constexpr int    QS_SZ = QK_K / 4;  // 64
    // CPU layout: [scales | qs | d(2) dmin(2)]
    static constexpr int    BYTES_PER_BLOCK = SCALES_SZ + QS_SZ + 4; // 84

    __device__ __forceinline__ uint16_t ld_u16_le(const uint8_t* p) {
        return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    }
}

// Kernel: writes __half
__global__ void k_deq_q2_k_to_f16_scales_qs_tail(
    const uint8_t* __restrict__ bytes,
    __half* __restrict__ out,
    size_t n_elems,
    int    blocks)
{
    using namespace q2k_detail;

    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;

    const size_t block_byte_off = static_cast<size_t>(b) * BYTES_PER_BLOCK;
    const uint8_t* base_ptr = bytes + block_byte_off;

    // [scales | qs | d(2) dmin(2)]
    const uint8_t* scales = base_ptr;                 // 16 bytes
    const uint8_t* qs = base_ptr + SCALES_SZ;     // 64 bytes
    const uint8_t* tail = qs + QS_SZ;               // 4 bytes

    const uint16_t d16 = q2k_detail::ld_u16_le(tail + 0);
    const uint16_t dmin16 = q2k_detail::ld_u16_le(tail + 2);

    const float d = __half2float(__ushort_as_half(d16));
    const float dmin = __half2float(__ushort_as_half(dmin16));

    const size_t out_base = static_cast<size_t>(b) * QK_K;
    if (out_base >= n_elems) return;
    const int block_elems = static_cast<int>(min(static_cast<size_t>(QK_K), n_elems - out_base));

#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const uint8_t s = scales[g];
        const float dl = d * float(s & 0x0F);
        const float ml = dmin * float(s >> 4);

        const uint8_t b0 = qs[g * 4 + 0];
        const uint8_t b1 = qs[g * 4 + 1];
        const uint8_t b2 = qs[g * 4 + 2];
        const uint8_t b3 = qs[g * 4 + 3];

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx_in_block = g * 16 + i;
            if (idx_in_block >= block_elems) break;

            const uint8_t pack = (i < 4) ? b0 : (i < 8) ? b1 : (i < 12) ? b2 : b3;
            const uint8_t shift = static_cast<uint8_t>((i & 3) * 2);
            const uint8_t q2 = static_cast<uint8_t>((pack >> shift) & 0x3u);

            // Match CPU: val = dl * q2 - ml
            const float val = dl * float(q2) - ml;

            out[out_base + idx_in_block] = __float2half_rn(val);
        }
    }
}


static __half* dequant_try_Q8_0(const uint8_t* bytes, size_t sz, size_t n_elems)
{
    const int blocks = int((n_elems + QK - 1) / QK);

    // Q8_0 => blocks * (2 + 32)
    size_t expect = size_t(blocks) * (2 + QK);
    if (sz == expect) {
        std::vector<int8_t>   hq(size_t(blocks) * QK);
        std::vector<uint16_t> hs(blocks);
        const uint8_t* p = bytes;
        for (int b = 0; b < blocks; b++) {
            uint16_t bits;
            memcpy(&bits, p, 2);
            p += 2;
            hs[b] = bits;
            memcpy(&hq[size_t(b) * QK], p, QK);
            p += QK;
        }
        int8_t* dq = nullptr;
        uint16_t* ds = nullptr;
        __half* out = nullptr;
        CUDA_OK(cudaMalloc(&dq, hq.size()));
        CUDA_OK(cudaMalloc(&ds, hs.size() * sizeof(uint16_t)));
        CUDA_OK(cudaMalloc(&out, n_elems * sizeof(__half)));
        CUDA_OK(cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(ds, hs.data(), hs.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        int th = 256, bl = (blocks + th - 1) / th;
        k_deq_q8 << <bl, th >> > (dq, ds, out, blocks);
        CUDA_OK(cudaDeviceSynchronize());
        cudaFree(dq);
        cudaFree(ds);
        return out;
    }

    return nullptr;
}
// Try to dequantize a quant payload by size signature; return device FP16 (caller cudaFree) or nullptr
static __half* dequant_try_Q2_K(const uint8_t* bytes, size_t sz, size_t n_elems)
{
    using namespace q2k_detail;

    if (!bytes || n_elems == 0) return nullptr;

    auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };
    const int    blocks = int(ceil_div(n_elems, size_t(QK_K)));
    const size_t expect = size_t(blocks) * BYTES_PER_BLOCK;
    if (sz != expect) {
        // Not a raw Q2_K payload (or wrong size for QK_K=256).
        return nullptr;
    }

    uint8_t* d_bytes = nullptr;
    __half* d_out = nullptr;

    CUDA_OK(cudaMalloc(&d_bytes, sz));
    CUDA_OK(cudaMemcpy(d_bytes, bytes, sz, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_out, n_elems * sizeof(__half)));

    const int th = 256;
    const int bl = (blocks + th - 1) / th;
    k_deq_q2_k_to_f16_scales_qs_tail << <bl, th >> > (d_bytes, d_out, n_elems, blocks);
    //CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    cudaFree(d_bytes);
    return d_out;  // device pointer; free with cudaFree
}

__global__ void k_f32_to_f16(__half* o, const float* x, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        o[i] = __float2half(x[i]);
}

// =============== Main ===============
int main4(int argc, char** argv)
{
    /*std::string pmp, pfp;
    bool report = false, diffHL = false;

    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--mpgguf" && i + 1 < argc)
            pmp = argv[++i];
        else if (a == "--fp16" && i + 1 < argc)
            pfp = argv[++i];
        else if (a == "--report")
            report = true;
        else if (a == "--diff-high-low")
            diffHL = true;
        else
        {
            std::cerr << "Usage: --mpgguf <file> --fp16 <f16_or_f32.gguf> [--report] [--diff-high-low]\n";
            return 1;
        }
    }
    if (pmp.empty() || pfp.empty())
    {
        std::cerr << "Missing --mpgguf or --fp16\n";
        return 1;
    }
    auto split = [](const std::string& s, char delimiter) -> std::vector<std::string>
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    };

    auto pfps = split(pfp, ',');*/

    std::string pmp;
    bool report = true, diffHL = true;
    //pmp = "Qwen3-1.7B.mpgguf";
    //std::vector<std::string> pfps = { "Qwen3-1.7B-BF16.gguf" };

    pmp = "Qwen3-30B-A3B.mpgguf";
    std::vector<std::string> pfps = { "Qwen3-30B-A3B-BF16-00001-of-00002.gguf", "Qwen3-30B-A3B-BF16-00002-of-00002.gguf" };

    MP mp;
    if (!load_mp(pmp, mp))
    {
        std::cerr << "bad mpgguf\n";
        return 1;
    }
    std::vector<std::shared_ptr<GGUFIndex>> ggs;
    ggs.resize(pfps.size());

    for (size_t i = 0; i < pfps.size(); i++)
        ggs[i] = parse_gguf_info(i, pfps[i]);

    // Map baseline by name
    std::unordered_map<std::string, TensorInfo*> truth;
    for (auto& gg : ggs)
    {
        for (auto& r : gg->tensors)
        {
            truth[r.name] = &r;
        }
    }

    auto N = [](const std::vector<uint64_t>& d) -> size_t {
        return numel(d);
        };
    const size_t kMaxElems = size_t(1) << 36;  // Defensive ~64B elements

    // Accumulators
    double s_low = 0, n_low = 0, s_high = 0, n_high = 0, s_hl = 0, n_hl = 0;
    float  m_low = 0, m_high = 0, m_hl = 0;
    size_t validated_high = 0, validated_low = 0;

    auto reduce = [&](const __half* A, const __half* B, size_t n, double& s, double& Nacc, float& m) {
        std::vector<__half> ha(n), hb(n);
        CUDA_OK(cudaMemcpy(ha.data(), A, n * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(hb.data(), B, n * sizeof(__half), cudaMemcpyDeviceToHost));
        double ss = 0;
        float  mx = 0;
        for (size_t i = 0; i < n; i++) {
            float d = __half2float(ha[i]) - __half2float(hb[i]);

            if (std::isnan(d) || std::isinf(d))
                continue;

            ss += double(d) * double(d);
            float ad = fabsf(d);
            if (ad > mx) {
                mx = ad;
            }
        }
        s += ss;
        Nacc += double(n);
        if (mx > m) {
            m = mx;
        }
        };

    for (const auto& r : mp.recs)
    {
        auto it = truth.find(r.name);
        if (it == truth.end())
        {
            if (report)
                std::cerr << "SKIP (missing in baseline): " << r.name << "\n";
            continue;
        }
        const TensorInfo& tr = *it->second;

        if (tr.dims != r.dims)
        {
            if (report)
                std::cerr << "SKIP (shape mismatch): " << r.name << "\n";
            continue;
        }

        size_t n = N(r.dims);
        if (n == 0 || n > kMaxElems)
        {
            if (report)
                std::cerr << "SKIP (absurd n=" << n << "): " << r.name << "\n";
            continue;
        }

        // Build FP16 truth (accept FP16 or FP32 tensor payloads)
        __half* d_truth = nullptr;
        auto stream_data = readStreamData(ggs[tr.splitId]->f, n * sizeof(size_t), tr.data_off);
        const uint8_t* tb = stream_data.data();
        if (tr.ggml_type == QuantizationType::BF16)
        {
            CUDA_OK(cudaMalloc(&d_truth, tr.data_sz));
            CUDA_OK(cudaMemcpy(d_truth, tb, tr.data_sz, cudaMemcpyHostToDevice));
        }
        else if (tr.ggml_type == QuantizationType::F32)
        {
            float* df = nullptr;
            CUDA_OK(cudaMalloc(&df, tr.data_sz));
            CUDA_OK(cudaMemcpy(df, tb, tr.data_sz, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMalloc(&d_truth, n * sizeof(__half)));
            int th = 256, bl = (int)((n + th - 1) / th);
            k_f32_to_f16 << <bl, th >> > (d_truth, df, n);
            CUDA_OK(cudaDeviceSynchronize());
            cudaFree(df);
        }
        else
        {
            // Unknown baseline tensor type; zero it so it doesn't pollute aggregates
            CUDA_OK(cudaMalloc(&d_truth, n * sizeof(__half)));
            CUDA_OK(cudaMemset(d_truth, 0, n * sizeof(__half)));
            if (report) {
                std::cerr << "WARN: baseline tensor " << r.name << " not FP16/FP32, treated as zeros\n";
            }
        }

        // HIGH (Q8_0 expected)
        if (r.flags & 2 && r.sz_high && in_range(r.off_high, r.sz_high, mp.data_sz) && r.g_high == QuantizationType::Q8_0)
        {
            auto data_high = readStreamData(mp.f, r.sz_high, mp.data_offset + r.off_high);
            const uint8_t* hb = data_high.data();
            
            __half* d_high = dequant_try_Q8_0(hb, (size_t)r.sz_high, n);
            if (d_high)
            {
                reduce(d_high, d_truth, n, s_high, n_high, m_high);
                cudaFree(d_high);
                validated_high++;
                std::cout << "Tensor name for INT8_0: " << validated_high << ", " << r.name << "\n";
            }
            else if (report)
            {
                std::cerr << "WARN: unrecognized HIGH payload for " << r.name << "\n";
            }
        }

        // LOW (legacy scalar 2-bit only; Q2_K/IQ2_* skipped)
        if (r.flags & 1 && r.sz_low && in_range(r.off_low, r.sz_low, mp.data_sz) && r.g_low == QuantizationType::Q2_K)
        {
            auto data_low = readStreamData(mp.f, r.sz_low, mp.data_offset + r.off_low);
            const uint8_t* lb = data_low.data();
            __half* d_low = dequant_try_Q2_K(lb, (size_t)r.sz_low, n);
            if (d_low)
            {
                reduce(d_low, d_truth, n, s_low, n_low, m_low);
                cudaFree(d_low);
                validated_low++;
                std::cout << "Tensor name for INT2_K: " << validated_low << ", " << r.name << "\n";
            }
            else if (report)
                std::cerr << "NOTE: skipping LOW dequant (likely Q2_K / IQ2_*): " << r.name << "\n";
        }

        cudaFree(d_truth);
    }

    auto outStats = [&](const char* tag, double s, double Nacc, float m, size_t ok) {
        if (Nacc <= 0) {
            std::cout << tag << ": N=0 (no tensors validated)\n";
            return;
        }
        double mse = s / Nacc;
        double rmse = std::sqrt(mse);
        std::cout << tag << ": tensors=" << ok << "  N=" << (uint64_t)Nacc << "  MSE=" << mse << "  RMSE=" << rmse
            << "  max =" << m << "\n";
        };

    outStats("HIGH(Q8_0) vs FP16", s_high, n_high, m_high, validated_high);
    outStats("LOW(2-bit)  vs FP16", s_low, n_low, m_low, validated_low);
    if (diffHL)
        outStats("HIGH vs LOW", s_hl, n_hl, m_hl, std::min(validated_high, validated_low));

    return 0;
}
