// mpgguf_validate_cpu.cpp — CPU-only version of MPGGUF v2 validator
// Build: g++ -O3 -std=c++17 -o mpgguf_validate_cpu mpgguf_validate_cpu.cpp
// Usage:
//   ./mpgguf_validate_cpu --mpgguf model.mpgguf --fp16 baseline_f16_or_f32.gguf [--report] [--diff-high-low]
//
// Notes:
//   * Mirrors the CUDA version’s logic but runs on CPU (no CUDA required).
//   * Dequantizes HIGH (Q8_0) and legacy 2-bit LOW (8B/16B code payloads) by size signature.
//   * Baseline GGUF reader tolerantly skips KV and accepts FP16 or FP32 tensor payloads.
//   * Computes MSE, RMSE, and max|Δ| vs FP16 baseline (and optional HIGH vs LOW).
//   * Unknown 2-bit layouts (Q2_K / IQ2_*) are skipped.
//

#include "gguf_parser.h"
#include "mpgguf_parser.h"
#include "parser_util.h"

using namespace monadd;

#ifndef QK
#    define QK 32  // scalar block size used by Q8_0 and the legacy 2-bit variants (must match container)
#endif

// https://www.abhik.xyz/articles/ggml-structure
// 
// ---------- Dequantizers (CPU) ----------

// Dequantizes Q8_0 payload (blocks × [float scale][int8 × QK]) into half
static std::vector<float> deq_q8_to_f16(const uint8_t * bytes, size_t sz, size_t n_elems) 
{
    const int blocks = int((n_elems + QK - 1) / QK);
    size_t    expect = size_t(blocks) * (2 + QK);
    if (sz != expect) {
        return {};
    }
    std::vector<float> out(n_elems);
    const uint8_t *  p = bytes;
    for (int b = 0; b < blocks; ++b) 
    {
        
        uint16_t bits;
        std::memcpy(&bits, p, 2);

        uint16_t h = static_cast<uint16_t>(p[0])
            | static_cast<uint16_t>(p[1]) << 8;

        float d = half::half_to_float(bits);

        if (isinf(d) || isnan(d))
            d = 0.0;

        p += 2;
        for (int i = 0; i < QK; ++i) {
            size_t idx = size_t(b) * QK + i;
            if (idx >= n_elems) 
                break;
            int8_t q = *(const int8_t *) (p + i);
            float  v = d * float(q);

            if (isinf(v) || isnan(v))
                d = 0.0;

            out[idx] = v;
        }
        p += QK;
    }
    return out;
}

// Q2_K → f16. Assumes bytes points to the raw tensor payload (no padding).
// Returns empty vector on size mismatch.
std::vector<float> deq_q2_k_to_f16(const uint8_t* bytes,
    size_t sz,
    size_t n_elems,
    int QK_K = 256) {
    if (QK_K % 16 != 0) {
        throw std::invalid_argument("QK_K must be divisible by 16");
    }
    if (n_elems % static_cast<size_t>(QK_K) != 0) {
        throw std::invalid_argument("n_elems must be a multiple of QK_K");
    }

    const size_t n_blocks = n_elems / static_cast<size_t>(QK_K);
    const size_t scales_sz = static_cast<size_t>(QK_K) / 16; // one byte per 16 values
    const size_t qs_sz = static_cast<size_t>(QK_K) / 4;  // 2 bits/value
    const size_t per_block = scales_sz + qs_sz + 4;          // +2 bytes d +2 bytes dmin

    const size_t expected_sz = n_blocks * per_block;
    if (sz < expected_sz) {
        throw std::invalid_argument("Input byte buffer is too small for the given n_elems/QK_K layout");
    }

    std::vector<float> out(n_elems);

    const uint8_t* p = bytes;

    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* scales = p;                 // length scales_sz
        const uint8_t* qs = p + scales_sz;     // length qs_sz
        const uint8_t* tail = qs + qs_sz;        // 4 bytes: d (2), dmin (2)

        // read fp16 values
        uint16_t d_bits = static_cast<uint16_t>(tail[0] | (tail[1] << 8));
        uint16_t dm_bits = static_cast<uint16_t>(tail[2] | (tail[3] << 8));
        float d = half::half_to_float(d_bits);
        float dmin = half::half_to_float(dm_bits);

        // For each group of 16 values
        // Each group consumes 1 scale byte and 4 data bytes (16 * 2 bits)
        const size_t groups = static_cast<size_t>(QK_K) / 16;
        const size_t block_out_base = b * static_cast<size_t>(QK_K);

        for (size_t g = 0; g < groups; ++g) {
            const uint8_t s = scales[g];
            const float dl = d * static_cast<float>(s & 0x0F);
            const float ml = dmin * static_cast<float>(s >> 4);

            // 4 bytes for 16 2-bit values
            const uint8_t b0 = qs[g * 4 + 0];
            const uint8_t b1 = qs[g * 4 + 1];
            const uint8_t b2 = qs[g * 4 + 2];
            const uint8_t b3 = qs[g * 4 + 3];

            // unpack in order: each byte holds 4 values (bits 0-1,2-3,4-5,6-7)
            const uint8_t pack[4] = { b0, b1, b2, b3 };
            for (size_t i = 0; i < 16; ++i) {
                const uint8_t byte = pack[i >> 2];          // i/4
                const uint8_t shift = static_cast<uint8_t>((i & 3) * 2); // (i%4)*2
                const uint8_t q2 = static_cast<uint8_t>((byte >> shift) & 0x3u);
                const float val = dl * static_cast<float>(q2) - ml;
                out[block_out_base + g * 16 + i] = val;
            }
        }

        p += per_block;
    }

    return out;
}

// Try to dequantize by size-signature
static std::vector<float> dequant_try(const uint8_t * bytes, size_t sz, size_t n_elems) {
    // Q8_0?
    {
        auto v = deq_q8_to_f16(bytes, sz, n_elems);
        if (!v.empty()) {
            return v;
        }
    }

    // Legacy scalar 2-bit?
    {
        auto v = deq_q2_k_to_f16(bytes, sz, n_elems);
        if (!v.empty()) {
            return v;
        }
    }
    
    return {};  // unknown layout
}

// Convert float array (baseline FP32) to half
static std::vector<bf16> f32_to_f16(const float * x, size_t n) {
    std::vector<bf16> o(n);
    for (size_t i = 0; i < n; i++) {
        o[i] = bf16(x[i]);
    }
    return o;
}

// ---------- Main ----------
int main3(int argc, char ** argv) {
    std::string pmp;
    //bool        report = false, diffHL = false;

    //for (int i = 1; i < argc; i++) {
    //    std::string a = argv[i];
    //    if (a == "--mpgguf" && i + 1 < argc) {
    //        pmp = argv[++i];
    //    } else if (a == "--fp16" && i + 1 < argc) {
    //        pfp = argv[++i];
    //    } else if (a == "--report") {
    //        report = true;
    //    } else if (a == "--diff-high-low") {
    //        diffHL = true;
    //    } else {
    //        std::cerr << "Usage: --mpgguf <file> --fp16 <f16_or_f32.gguf> [--report] [--diff-high-low]\n";
    //        return 1;
    //    }
    //}

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

    for (auto& gg:ggs)
    {
        for (auto& r : gg->tensors) 
        {
            truth[r.name] = &r;
        }
    }

    auto N = [](const std::vector<uint64_t> & d) -> size_t 
    {
        return numel(d);
    };
    const size_t kMaxElems = size_t(1) << 36;  // defensive

    // Accumulators
    long double s_low = 0, n_low = 0, s_high = 0, n_high = 0, s_hl = 0, n_hl = 0;
    float       m_low = 0, m_high = 0, m_hl = 0;
    size_t      validated_high = 0, validated_low = 0;
    std::vector<long double> vHighs;

    auto reduce = [&](const std::vector<float> & A, const std::vector<float> & B, size_t n, long double & s,
                      long double & Nacc, float & m) {
        long double ss = 0;
        float       mx = 0;
        for (size_t i = 0; i < n; i++) 
        {
            float da = (float) A[i];
            float db = (float) B[i];
            float d  = da - db;

            if (isnan(d) || isinf(d))
                continue;

            ss += (long double) d * (long double) d;
            float ad = std::fabs(d);
            if (ad > mx) {
                mx = ad;
            }
        }
        s += ss;
        Nacc += (long double) n;
        if (mx > m) {
            m = mx;
        }
    };

    for (const auto & r : mp.recs) {
        auto it = truth.find(r.name);
        if (it == truth.end()) {
            if (report) {
                std::cerr << "SKIP (missing in baseline): " << r.name << "\n";
            }
            continue;
        }
        const TensorInfo & tr = *it->second;

        if (tr.dims != r.dims) {
            if (report) {
                std::cerr << "SKIP (shape mismatch): " << r.name << "\n";
            }
            continue;
        }

        size_t n = N(r.dims);
        if (n == 0 || n > kMaxElems) {
            if (report) {
                std::cerr << "SKIP (absurd n=" << n << "): " << r.name << "\n";
            }
            continue;
        }

        // Build FP16 truth (accept FP16 or FP32 tensor payloads)
        std::vector<bf16> h_truth2(n, bf16(float(0.0)));
        auto stream_data = readStreamData(ggs[tr.splitId]->f, n * sizeof(size_t), tr.data_off);
        const uint8_t* tb = stream_data.data();
        if (tr.ggml_type == QuantizationType::BF16)
        {
            // baseline FP16
            for (size_t i = 0; i < n; i++) {
                uint16_t bits;
                std::memcpy(&bits, tb + i * 2, 2);
                h_truth2[i] = bf16(bits);
            }
        } 
        else if (tr.ggml_type == QuantizationType::F32)
        {
            // baseline FP32
            const float * fp = reinterpret_cast<const float *>(tb);
            h_truth2          = f32_to_f16(fp, n);
        }
        else 
        {
            // Unknown baseline tensor type; treat as zeros (warn if report)
            if (report) {
                std::cerr << "WARN: baseline tensor " << r.name << " not FP16/FP32, treated as zeros\n";
            }
        }

        std::vector<float> f32_truths2;
        f32_truths2.reserve(h_truth2.size());
        for (size_t i = 0; i < h_truth2.size(); i++)
            f32_truths2.push_back(h_truth2[i].operator float());

        // LOW (legacy scalar 2-bit only)
        if (r.flags & 1 && r.sz_low && in_range(r.off_low, r.sz_low, mp.data_sz) && r.g_low == QuantizationType::Q2_K)
        {
            auto data_low = readStreamData(mp.f, r.sz_low, mp.data_offset + r.off_low);
            const uint8_t* lb = data_low.data();
            auto            d_low = deq_q2_k_to_f16(lb, (size_t) r.sz_low, n);
            if (!d_low.empty()) 
            {
                reduce(d_low, f32_truths2, n, s_low, n_low, m_low);
                validated_low++;

                std::cout << "Tensor name for INT2_K: " << validated_low << ", " << r.name << "\n";
            } 
            else if (report) 
                std::cerr << "NOTE: skipping LOW dequant (likely Q2_K / IQ2_*): " << r.name << "\n";
        }

        // HIGH (Q8_0 expected)
        if (r.flags & 2 && r.sz_high && in_range(r.off_high, r.sz_high, mp.data_sz) && r.g_high == QuantizationType::Q8_0)
        {
            auto data_high = readStreamData(mp.f, r.sz_high, mp.data_offset + r.off_high);
            const uint8_t* hb = data_high.data();
            //auto            d_high = dequant_try(hb, (size_t) r.sz_high, n);
            auto            d_high = deq_q8_to_f16(hb, (size_t)r.sz_high, n);
            if (!d_high.empty()) {
                reduce(d_high, f32_truths2, n, s_high, n_high, m_high);
                validated_high++;
                std::cout << "Tensor name for INT8_0: " << validated_high << ", " << r.name << "\n";
            }
            else if (report) {
                std::cerr << "WARN: unrecognized HIGH payload for " << r.name << "\n";
            }
        }
    }

    auto outStats = [&](const char * tag, long double s, long double Nacc, float m, size_t ok) {
        if (Nacc <= 0) {
            std::cout << tag << ": N=0 (no tensors validated)\n";
            return;
        }
        long double mse  = s / Nacc;
        long double rmse = std::sqrt((long double) mse);
        std::cout << tag << ": tensors=" << ok << "  N=" << (unsigned long long) Nacc << "  MSE=" << (double) mse
                  << "  RMSE=" << (double) rmse << "  max =" << m << "\n";
    };

    outStats("HIGH(Q8_0) vs FP16", s_high, n_high, m_high, validated_high);
    outStats("LOW(2-bit)  vs FP16", s_low, n_low, m_low, validated_low);
    //if (diffHL) {
    //    outStats("HIGH vs LOW", s_hl, n_hl, m_hl, std::min(validated_high, validated_low));
    //}
    return 0;
}
