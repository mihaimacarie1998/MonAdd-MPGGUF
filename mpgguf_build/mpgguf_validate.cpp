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
// Differences from CUDA version:
//   * Uses a minimal IEEE 754 half <-> float converter (software).
//   * All “device” buffers are std::vector on host.
//   * Small bug fixed in load_mp(): header check now returns false only when invalid.

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef QK
#    define QK 32  // scalar block size used by Q8_0 and the legacy 2-bit variants (must match container)
#endif

// https://www.abhik.xyz/articles/ggml-structure
// 
// ---------- helpers ----------

// ---------- GGUF baseline reader (tolerant KV skipper) ----------
enum {
    GGUF_U8 = 0,
    GGUF_I8 = 1,
    GGUF_U16 = 2,
    GGUF_I16 = 3,
    GGUF_U32 = 4,
    GGUF_I32 = 5,
    GGUF_F32 = 6,
    GGUF_BOOL = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9,
    GGUF_U64 = 10,
    GGUF_I64 = 11,
    GGUF_F64 = 12,
    GGUF_BYTES = 13
};

static int scalar_size(int t) {
    switch (t) {
    case GGUF_U8:
    case GGUF_I8:
    case GGUF_BOOL:
        return 1;
    case GGUF_U16:
    case GGUF_I16:
        return 2;
    case GGUF_U32:
    case GGUF_I32:
    case GGUF_F32:
        return 4;
    case GGUF_U64:
    case GGUF_I64:
    case GGUF_F64:
        return 8;
    default:
        return -1;
    }
}

enum
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39
};

struct bf16
{
    uint16_t bits;

    bf16() : bits(0) {}
    bf16(uint16_t f) : bits(f) {}
    bf16(float f) { bits = float_to_half(f); }
    operator float() const { return half_to_float(bits); }

    static uint16_t float_to_half(float f) 
    {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        // Round to nearest even
        uint32_t lsb = (x >> 16) & 1;
        uint32_t rounding_bias = 0x7FFF + lsb;
        x += rounding_bias;
        return static_cast<uint16_t>(x >> 16);
    }

    static float half_to_float(uint16_t h) 
    {
        uint32_t x = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &x, sizeof(f));
        return f;
    }
};

/// --- Simple FP16 type (IEEE 754 half) ---
struct half 
{
    uint16_t bits;

    half() : bits(0) {}
    half(uint16_t f) : bits(f) {}
    half(float f) { bits = float_to_half(f); }
    operator float() const { return half_to_float(bits); }

    static uint16_t float_to_half(float f) 
    {
        uint32_t x = *reinterpret_cast<uint32_t*>(&f);
        uint16_t sign = (x >> 16) & 0x8000;
        uint32_t mantissa = x & 0x7fffff;
        int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
        if (exp <= 0) {
            if (exp < -10) return sign;
            mantissa = (mantissa | 0x800000) >> (1 - exp);
            return sign | (mantissa + 0x1000) >> 13;
        }
        else if (exp >= 31) {
            return sign | 0x7c00;
        }
        return sign | (exp << 10) | ((mantissa + 0x1000) >> 13);
    }

    static float half_to_float(uint16_t h) 
    {
        uint32_t sign = static_cast<uint32_t>((h >> 15) & 0x1u);
        uint32_t exp = static_cast<uint32_t>((h >> 10) & 0x1Fu);
        uint32_t mantissa = static_cast<uint32_t>(h & 0x3FFu);

        uint32_t f; // float32 bit pattern

        if (exp == 0) {
            if (mantissa == 0) {
                // +/- 0.0
                f = (sign << 31);
            }
            else {
                // Subnormal half: normalize mantissa
                uint32_t e = 1; // effective exponent for half subnormals starts at 1
                while ((mantissa & 0x400u) == 0u) { // until leading 1 appears at bit 10
                    mantissa <<= 1;
                    --e;
                }
                mantissa &= 0x3FFu; // drop the leading 1
                // Convert exponent bias: (e - 1) + (127 - 15)
                uint32_t exp32 = static_cast<uint32_t>(static_cast<int32_t>(e) - 1 + 127 - 15);
                f = (sign << 31) | (exp32 << 23) | (mantissa << 13);
            }
        }
        else if (exp == 0x1Fu) {
            // Inf / NaN
            f = (sign << 31) | 0x7F800000u | (mantissa << 13);
        }
        else {
            // Normalized
            uint32_t exp32 = exp + (127 - 15);
            f = (sign << 31) | (exp32 << 23) | (mantissa << 13);
        }

        float out;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }
};

/// --- Half16 vector (CPU equivalent of __half16) ---
struct half16
{
    half data[16];

    half16() { for (auto& x : data) x = half(0.0f); }

    explicit half16(float val) {
        for (auto& x : data) x = half(val);
    }

    explicit half16(const float* vals) {
        for (int i = 0; i < 16; ++i)
            data[i] = half(vals[i]);
    }

    float get(int i) const { return static_cast<float>(data[i]); }
    void set(int i, float val) { data[i] = half(val); }

    half& operator[](int i) { return data[i]; }
    const half& operator[](int i) const { return data[i]; }

    // --- Basic arithmetic operations ---
    half16 operator+(const half16& other) const {
        half16 result;
        for (int i = 0; i < 16; ++i)
            result.data[i] = half(static_cast<float>(data[i]) + static_cast<float>(other.data[i]));
        return result;
    }

    half16 operator-(const half16& other) const {
        half16 result;
        for (int i = 0; i < 16; ++i)
            result.data[i] = half(static_cast<float>(data[i]) - static_cast<float>(other.data[i]));
        return result;
    }

    half16 operator*(const half16& other) const {
        half16 result;
        for (int i = 0; i < 16; ++i)
            result.data[i] = half(static_cast<float>(data[i]) * static_cast<float>(other.data[i]));
        return result;
    }

    half16 operator/(const half16& other) const {
        half16 result;
        for (int i = 0; i < 16; ++i)
            result.data[i] = half(static_cast<float>(data[i]) / static_cast<float>(other.data[i]));
        return result;
    }
};

static inline bool aligned32(uint64_t x) 
{
    return (x & 31ull) == 0ull;
}

static inline bool aligned64(uint64_t x) 
{
    return (x & 63ull) == 0ull;
}

static uint64_t align_up(uint64_t off, uint32_t align) 
{
    const uint64_t mask = static_cast<uint64_t>(align) - 1u;
    return (off + mask) & ~mask;
}

static inline bool is_ascii_identifier(const std::string & s) {
    if (s.empty() || s.size() > 512) {
        return false;
    }
    for (unsigned char c : s) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
              c == '/' || c == '-')) {
            return false;
        }
    }
    return true;
}

static inline uint32_t rd_u32(const uint8_t * p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t rd_u64(const uint8_t * p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static bool in_range(uint64_t off, uint64_t sz, size_t total) {
    return off <= (uint64_t) total && sz <= (uint64_t) total && (off + sz) <= (uint64_t) total;
}

static size_t numel(const std::vector<uint64_t> & d) {
    size_t n = 1;
    for (auto v : d) {
        n *= size_t(v);
    }
    return n;
}

// ---------- MPGGUF v2 reader ----------
struct Rec {
    std::string           name;
    uint32_t              nd;
    std::vector<uint64_t> dims;
    uint32_t              flags, g_low, g_high, g_fp;
    uint64_t              off_low, sz_low, off_high, sz_high, off_fp, sz_fp;
};

struct MP {
    std::vector<Rec>     recs;
    std::vector<uint8_t> kv;
    std::vector<uint8_t> data;
};

bool load_mp(const std::string & path, MP & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "open failed: " << path << "\n";
        return false;
    }

    // Header
    char mg[7];
    f.read(mg, 7);
    uint8_t ver = 0;
    f.read((char *) &ver, 1);
    if (std::string(mg, 7) != "MPGGUF2" || ver != 2) {
        std::cerr << "Not MPGGUF2 (" << path << ")\n";
        return false;  // fixed from original snippet
    }

    uint64_t kvsz = 0;
    uint32_t nt   = 0;
    f.read((char *) &kvsz, 8);
    f.read((char *) &nt, 4);

    const uint64_t kMaxT = 200000;
    if (nt == 0 || nt > kMaxT) {
        std::cerr << "ERROR: suspicious mpgguf n_t=" << nt << "\n";
        return false;
    }

    // Directory
    out.recs.reserve(nt);
    const uint32_t kMaxName = 4096;
    for (uint32_t i = 0; i < nt; i++) {
        uint32_t nl = 0;
        f.read((char *) &nl, 4);
        if (nl == 0 || nl > kMaxName) {
            std::cerr << "ERROR: mpgguf name length " << nl << "\n";
            return false;
        }
        std::string name(nl, '\0');
        f.read(name.data(), nl);
        if (!is_ascii_identifier(name)) {
            std::cerr << "ERROR: mpgguf bad name " << name << "\n";
            return false;
        }

        uint32_t nd = 0;
        f.read((char *) &nd, 4);
        if (nd == 0 || nd > 6) {
            std::cerr << "ERROR: mpgguf bad nd=" << nd << " for " << name << "\n";
            return false;
        }
        std::vector<uint64_t> dims(nd);
        for (uint32_t d = 0; d < nd; ++d) {
            f.read((char *) &dims[d], 8);
            if (dims[d] == 0 || dims[d] > (uint64_t) 1e10) {
                std::cerr << "ERROR: mpgguf bad dim[" << d << "]=" << dims[d] << " for " << name << "\n";
                return false;
            }
        }

        uint32_t flags = 0, gL = 0, gH = 0, gF = 0;
        f.read((char *) &flags, 4);
        f.read((char *) &gL, 4);
        f.read((char *) &gH, 4);
        f.read((char *) &gF, 4);
        if ((flags & ~0x7u) != 0) {
            std::cerr << "ERROR: mpgguf flags reserved bits set for " << name << "\n";
            return false;
        }

        uint64_t oL = 0, sL = 0, oH = 0, sH = 0, oF = 0, sF = 0;
        f.read((char *) &oL, 8);
        f.read((char *) &sL, 8);
        f.read((char *) &oH, 8);
        f.read((char *) &sH, 8);
        f.read((char *) &oF, 8);
        f.read((char *) &sF, 8);

        if ((flags & 0x1) && !aligned64(oL)) {
            std::cerr << "ERROR: LOW off not 64B aligned\n";
            return false;
        }
        if ((flags & 0x2) && !aligned64(oH)) {
            std::cerr << "ERROR: HIGH off not 64B aligned\n";
            return false;
        }
        if ((flags & 0x4) && !aligned64(oF)) {
            std::cerr << "ERROR: FP off not 64B aligned\n";
            return false;
        }

        out.recs.push_back({ std::move(name), nd, std::move(dims), flags, gL, gH, gF, oL, sL, oH, sH, oF, sF });
    }

    // KV
    out.kv.resize(kvsz);
    if (kvsz) {
        f.read((char *) out.kv.data(), kvsz);
    }

    // Data region
    // 1. Get the current position
    std::streampos current_pos = f.tellg();
    current_pos = align_up(current_pos, 64);

    // 2. Seek to the end
    f.seekg(0, std::ios::end);

    // 3. Get the end position
    std::streampos end_pos = f.tellg();

    // 4. Calculate remaining size
    std::streamsize remaining_size = end_pos - current_pos;

    // 5. Seek back to the current position
    f.seekg(current_pos);
    
    // 6. Allocate and read
    out.data.resize(remaining_size);
    f.read((char *)& out.data[0], remaining_size);

    //out.data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    const size_t D = out.data.size();

    // Range checks
    for (const auto & r : out.recs) {
        if (r.sz_low && !in_range(r.off_low, r.sz_low, D)) {
            std::cerr << "ERROR: LOW slice OOB for " << r.name << "\n";
            return false;
        }
        if (r.sz_high && !in_range(r.off_high, r.sz_high, D)) {
            std::cerr << "ERROR: HIGH slice OOB for " << r.name << "\n";
            return false;
        }
        if (r.sz_fp && !in_range(r.off_fp, r.sz_fp, D)) {
            std::cerr << "ERROR: FP slice OOB for " << r.name << "\n";
            return false;
        }
    }
    return true;
}

static std::string rd_s(std::istream & f) {
    uint64_t n = 0;
    f.read((char *) &n, 8);
    const uint64_t kMaxStr = 4096ull * 1024ull;
    if (n > kMaxStr) {
        throw std::runtime_error("gguf string length over cap");
    }
    std::string s((size_t) n, '\0');
    if (n) {
        f.read(s.data(), (std::streamsize) n);
    }
    return s;
}

static void skip_bytes(std::istream & f) {
    uint64_t n = 0;
    f.read((char *) &n, 8);
    f.seekg((std::streamoff) n, std::ios::cur);
}

static void skipv(std::istream & f) {
    uint32_t t = 0;
    f.read((char *) &t, 4);
    int sz = scalar_size((int) t);
    if (sz > 0) {
        f.seekg(sz, std::ios::cur);
        return;
    }
    if (t == GGUF_STRING) {
        (void) rd_s(f);
        return;
    }
    if (t == GGUF_BYTES) {
        skip_bytes(f);
        return;
    }
    if (t == GGUF_ARRAY) {
        uint32_t et  = 0;
        uint64_t cnt = 0;
        f.read((char *) &et, 4);
        f.read((char *) &cnt, 8);
        int es = scalar_size((int) et);
        if (es > 0) {
            f.seekg((std::streamoff) es * (std::streamoff) cnt, std::ios::cur);
            return;
        }
        if (et == GGUF_STRING) {
            for (uint64_t i = 0; i < cnt; i++) {
                (void) rd_s(f);
            }
            return;
        }
        if (et == GGUF_BYTES) {
            for (uint64_t i = 0; i < cnt; i++) {
                skip_bytes(f);
            }
            return;
        }
        if (et == GGUF_ARRAY) {
            for (uint64_t i = 0; i < cnt; i++) {
                skipv(f);
            }
            return;
        }
        for (uint64_t i = 0; i < cnt; i++) {
            skip_bytes(f);  // fallback
        }
        return;
    }
    skip_bytes(f);  // unknown -> treat as BYTES
}

struct GRec {
    std::string           name;
    uint32_t              nd;
    std::vector<uint64_t> dims;
    uint32_t              g;
    uint64_t              off, sz;
};

struct GG {
    std::vector<GRec>    recs;
    std::vector<uint8_t> whole;
};

bool load_fp(const std::string & path, GG & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "open failed: " << path << "\n";
        return false;
    }

    char mg[4];
    f.read(mg, 4);
    if (std::string(mg, 4) != "GGUF") {
        std::cerr << "not GGUF: " << path << "\n";
        return false;
    }
    uint32_t ver = 0;
    f.read((char *) &ver, 4);
    uint64_t n_t = 0, n_kv = 0;
    f.read((char *) &n_t, 8);
    f.read((char *) &n_kv, 8);

    const uint64_t kMaxKV = 200000, kMaxT = 200000;
    if (n_kv == 0 || n_kv > kMaxKV) {
        std::cerr << "ERROR: suspicious n_kv=" << n_kv << "\n";
        return false;
    }
    if (n_t == 0 || n_t > kMaxT) {
        std::cerr << "ERROR: suspicious n_t=" << n_t << "\n";
        return false;
    }

    // Skip KV
    try {
        for (uint64_t i = 0; i < n_kv; i++) {
            (void) rd_s(f);
            skipv(f);
        }
    } catch (const std::exception & e) {
        std::cerr << "WARN: KV skip failed: " << e.what() << ", continuing\n";
    }

    // Tensor table
    std::vector<GRec> recs;
    recs.reserve((size_t) n_t);
    for (uint64_t i = 0; i < n_t; i++) {
        std::string name = rd_s(f);
        if (!is_ascii_identifier(name)) {
            std::cerr << "ERROR: bad tensor name " << name << "\n";
            return false;
        }

        uint32_t nd = 0;
        f.read((char *) &nd, 4);
        if (nd == 0 || nd > 6) {
            std::cerr << "ERROR: bad nd=" << nd << " for " << name << "\n";
            return false;
        }

        std::vector<uint64_t> dims(nd);
        for (uint32_t d = 0; d < nd; ++d) {
            f.read((char *) &dims[d], 8);
            if (dims[d] == 0 || dims[d] > (uint64_t) 1e10) {
                std::cerr << "ERROR: bad dim[" << d << "]=" << dims[d] << " for " << name << "\n";
                return false;
            }
        }

        uint32_t g = 0;
        f.read((char *) &g, 4);
        uint64_t off = 0;
        f.read((char *) &off, 8);
        if (!aligned32(off)) {
            std::cerr << "ERROR: tensor data offset not 32B aligned for " << name << "\n";
            return false;
        }

        recs.push_back({ std::move(name), nd, std::move(dims), g, off, 0 });
    }

    // 4) Remember the start of the data block
    uint64_t data_block_start = static_cast<uint64_t>(f.tellg());
    data_block_start = align_up(data_block_start, 32);

    // Sizes by next off / EOF
    f.seekg(0, std::ios::end);
    size_t fsz = (size_t) f.tellg();

    std::vector<GRec *> ord;
    ord.reserve(recs.size());
    for (auto & r : recs) {
        ord.push_back(&r);
    }
    std::sort(ord.begin(), ord.end(), [](auto * a, auto * b) { return a->off < b->off; });
    for (size_t i = 0; i < ord.size(); ++i) {
        uint64_t nxt = (i + 1 < ord.size()) ? ord[i + 1]->off : (uint64_t) fsz;
        if (nxt < ord[i]->off) {
            std::cerr << "ERROR: decreasing offsets in baseline\n";
            return false;
        }
        ord[i]->sz = nxt - ord[i]->off;
    }

    for (auto& r : recs)
        r.off += data_block_start;

    // Whole file
    f.seekg(0, std::ios::end);
    std::streampos file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    out.whole.resize(static_cast<size_t>(file_size));  // Allocate exact size

    f.read(reinterpret_cast<char *>(out.whole.data()), file_size);  // Read directly into buffer
    out.recs = std::move(recs);
    return true;
}

// ---------- Dequantizers (CPU) ----------
static const int8_t LUT2[4] = { -2, -1, 0, 1 };

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
    for (int b = 0; b < blocks; ++b) {
        
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
            if (idx >= n_elems) {
                break;
            }
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
int main(int argc, char ** argv) {
    std::string pmp, pfp;
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
    //pfp = "Qwen3-1.7B-BF16.gguf";

    pmp = "Qwen3-30B-A3B.mpgguf";
    pfp = "Qwen3-1.7B-BF16.gguf";

    if (pmp.empty() || pfp.empty()) {
        std::cerr << "Missing --mpgguf or --fp16\n";
        return 1;
    }

    MP mp;
    if (!load_mp(pmp, mp)) {
        std::cerr << "bad mpgguf\n";
        return 1;
    }
    GG gg;
    if (!load_fp(pfp, gg)) {
        std::cerr << "bad gguf baseline\n";
        return 1;
    }

    // Map baseline by name
    std::unordered_map<std::string, GRec *> truth;
    for (auto & r : gg.recs) {
        truth[r.name] = &r;
    }

    auto N = [](const std::vector<uint64_t> & d) -> size_t {
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
        for (size_t i = 0; i < n; i++) {
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
        const GRec & tr = *it->second;

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
        const uint8_t *  tb = gg.whole.data() + tr.off;
        if (tr.g == 30) {
            // baseline FP16
            for (size_t i = 0; i < n; i++) {
                uint16_t bits;
                std::memcpy(&bits, tb + i * 2, 2);
                h_truth2[i] = bf16(bits);
            }
        } else if (tr.g == 0) {
            // baseline FP32
            const float * fp = reinterpret_cast<const float *>(tb);
            h_truth2          = f32_to_f16(fp, n);
        } else {
            // Unknown baseline tensor type; treat as zeros (warn if report)
            if (report) {
                std::cerr << "WARN: baseline tensor " << r.name << " not FP16/FP32, treated as zeros\n";
            }
        }

        std::vector<float> f32_truths2;
        f32_truths2.reserve(h_truth2.size());
        for (size_t i = 0; i < h_truth2.size(); i++)
        {
            f32_truths2.push_back(h_truth2[i].operator float());
        }

        // LOW (legacy scalar 2-bit only)
        if (r.sz_low && in_range(r.off_low, r.sz_low, mp.data.size()) && r.g_low == 10)
        {
            const uint8_t * lb    = mp.data.data() + r.off_low;
            auto            d_low = deq_q2_k_to_f16(lb, (size_t) r.sz_low, n);
            if (!d_low.empty()) {
                reduce(d_low, f32_truths2, n, s_low, n_low, m_low);
                validated_low++;
            } else if (report) {
                std::cerr << "NOTE: skipping LOW dequant (likely Q2_K / IQ2_*): " << r.name << "\n";
            }
        }

        // HIGH (Q8_0 expected)
        if (r.sz_high && in_range(r.off_high, r.sz_high, mp.data.size()) && r.g_high == 8)
        {
            const uint8_t* hb = mp.data.data() + r.off_high;
            //auto            d_high = dequant_try(hb, (size_t) r.sz_high, n);
            auto            d_high = deq_q8_to_f16(hb, (size_t)r.sz_high, n);
            if (!d_high.empty()) {
                reduce(d_high, f32_truths2, n, s_high, n_high, m_high);
                validated_high++;
            }
            else if (report) {
                std::cerr << "WARN: unrecognized HIGH payload for " << r.name << "\n";
            }
        }

        // Optional HIGH vs LOW
        //if (diffHL && r.sz_high && r.sz_low && in_range(r.off_high, r.sz_high, mp.data.size()) &&
        //    in_range(r.off_low, r.sz_low, mp.data.size())) {
        //    const uint8_t * hb     = mp.data.data() + r.off_high;
        //    const uint8_t * lb     = mp.data.data() + r.off_low;
        //    auto            d_high = dequant_try(hb, (size_t) r.sz_high, n);
        //    auto            d_low  = dequant_try(lb, (size_t) r.sz_low, n);
        //    if (!d_high.empty() && !d_low.empty()) {
        //        reduce(d_high, d_low, n, s_hl, n_hl, m_hl);
        //    }
        //}
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
