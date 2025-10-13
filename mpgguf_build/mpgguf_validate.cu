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
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// ------------- CUDA helpers -------------
#define CUDA_OK(x)                                                                                          \
    do {                                                                                                    \
        auto _e = (x);                                                                                      \
        if (_e != cudaSuccess) {                                                                            \
            std::cerr << "CUDA " << cudaGetErrorString(_e) << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

#ifndef QK
#    define QK 32  // scalar block size used by Q8_0 and the legacy 2-bit variants
#endif

static inline bool aligned32(uint64_t x) {
    return (x & 31ull) == 0ull;
}

static inline bool aligned64(uint64_t x) {
    return (x & 63ull) == 0ull;
}

static inline bool is_ascii_identifier(const std::string & s) {
    if (s.empty() || s.size() > 512)
        return false;

    for (unsigned char c : s)
    {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
              c == '/' || c == '-')) {
            return false;
        }
    }
    return true;
}

//=============== MPGGUF v2 reader ===============
struct Rec
{
    std::string           name;
    uint32_t              nd;
    std::vector<uint64_t> dims;
    uint32_t              flags, g_low, g_high, g_fp;
    uint64_t              off_low, sz_low, off_high, sz_high, off_fp, sz_fp;
};

struct MP
{
    std::vector<Rec>     recs;
    std::vector<uint8_t> kv;
    std::vector<uint8_t> data;
};

static inline uint32_t rd_u32(const uint8_t * p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t rd_u64(const uint8_t * p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static bool in_range(uint64_t off, uint64_t sz, size_t total)
{
    return off <= (uint64_t) total && sz <= (uint64_t) total && (off + sz) <= (uint64_t) total;
}

bool load_mp(const std::string & path, MP & out)
{
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
        return false;
    }

    uint64_t kvsz = 0;
    uint32_t nt   = 0;
    f.read((char *) &kvsz, 8);
    f.read((char *) &nt, 4);

    // Sanity cap on tensor count
    const uint64_t kMaxT = 200000;
    if (nt == 0 || nt > kMaxT)
    {
        std::cerr << "ERROR: suspicious mpgguf n_t=" << nt << "\n";
        return false;
    }

    // Directory
    out.recs.reserve(nt);
    const uint32_t kMaxName = 4096;
    for (uint32_t i = 0; i < nt; i++)
    {
        uint32_t nl = 0;
        f.read((char *) &nl, 4);
        if (nl == 0 || nl > kMaxName)
        {
            std::cerr << "ERROR: mpgguf name length " << nl << "\n";
            return false;
        }
        std::string name(nl, '\0');
        f.read(name.data(), nl);
        if (!is_ascii_identifier(name))
        {
            std::cerr << "ERROR: mpgguf bad name " << name << "\n";
            return false;
        }

        uint32_t nd = 0;
        f.read((char *) &nd, 4);
        if (nd == 0 || nd > 6)
        {
            std::cerr << "ERROR: mpgguf bad nd=" << nd << " for " << name << "\n";
            return false;
        }
        std::vector<uint64_t> dims(nd);
        for (uint32_t d = 0; d < nd; ++d)
        {
            f.read((char *) &dims[d], 8);
            if (dims[d] == 0 || dims[d] > (uint64_t) 1e10)
            {
                std::cerr << "ERROR: mpgguf bad dim[" << d << "]=" << dims[d] << " for " << name << "\n";
                return false;
            }
        }

        uint32_t flags = 0, gL = 0, gH = 0, gF = 0;
        f.read((char *) &flags, 4);
        f.read((char *) &gL, 4);
        f.read((char *) &gH, 4);
        f.read((char *) &gF, 4);
        if ((flags & ~0x7u) != 0)
        {
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

        if (flags & 0x1)
        {
            if (!aligned64(oL))
            {
                std::cerr << "ERROR: LOW off not 64B aligned\n";
                return false;
            }
        }
        if (flags & 0x2)
        {
            if (!aligned64(oH)) {
                std::cerr << "ERROR: HIGH off not 64B aligned\n";
                return false;
            }
        }
        if (flags & 0x4)
        {
            if (!aligned64(oF)) {
                std::cerr << "ERROR: FP off not 64B aligned\n";
                return false;
            }
        }

        out.recs.push_back({ std::move(name), nd, std::move(dims), flags, gL, gH, gF, oL, sL, oH, sH, oF, sF });
    }

    // KV
    out.kv.resize(kvsz);
    if (kvsz)
    {
        f.read((char *) out.kv.data(), kvsz);
    }

// Data region
    // 1. Get the current position
    std::streampos current_pos = f.tellg();

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
    f.read((char *) &out.data[0], remaining_size);

    //out.data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    const size_t D = out.data.size();

    // Range checks now that data size is known
    for (const auto & r : out.recs)
    {
        if (r.sz_low && !in_range(r.off_low, r.sz_low, D))
        {
            std::cerr << "ERROR: LOW slice OOB for " << r.name << "\n";
            return false;
        }
        if (r.sz_high && !in_range(r.off_high, r.sz_high, D))
        {
            std::cerr << "ERROR: HIGH slice OOB for " << r.name << "\n";
            return false;
        }
        if (r.sz_fp && !in_range(r.off_fp, r.sz_fp, D))
        {
            std::cerr << "ERROR: FP slice OOB for " << r.name << "\n";
            return false;
        }
    }
    return true;
}

// =============== GGUF (baseline) reader ===============
enum
{
    GGUF_U8     = 0,
    GGUF_I8     = 1,
    GGUF_U16    = 2,
    GGUF_I16    = 3,
    GGUF_U32    = 4,
    GGUF_I32    = 5,
    GGUF_F32    = 6,
    GGUF_BOOL   = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY  = 9,
    GGUF_U64    = 10,
    GGUF_I64    = 11,
    GGUF_F64    = 12,
    GGUF_BYTES  = 13
};

static int scalar_size(int t)
{
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

static std::string rd_s(std::istream & f)
{
    uint64_t n = 0;
    f.read((char *) &n, 8);
    const uint64_t kMaxStr = 4096 * 1024;  // cap for safety
    if (n > kMaxStr) {
        throw std::runtime_error("gguf string length over cap");
    }
    std::string s((size_t) n, '\0');
    if (n) {
        f.read(s.data(), (std::streamsize) n);
    }
    return s;
}

static void skip_bytes(std::istream & f)
{
    uint64_t n = 0;
    f.read((char *) &n, 8);
    f.seekg((std::streamoff) n, std::ios::cur);
}

static void skipv(std::istream & f)
{
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
    skip_bytes(f);  // unknown top-level -> treat as BYTES
}

struct GRec
{
    std::string           name;
    uint32_t              nd;
    std::vector<uint64_t> dims;
    uint32_t              g;
    uint64_t              off, sz;
};

struct GG
{
    std::vector<GRec>    recs;
    std::vector<uint8_t> whole;
};

static size_t numel(const std::vector<uint64_t> & d)
{
    size_t n = 1;
    for (auto v : d) {
        n *= size_t(v);
    }
    return n;
}

bool load_fp(const std::string & path, GG & out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "open failed: " << path << "\n";
        return false;
    }

    // Header
    char mg[4];
    f.read(mg, 4);
    if (std::string(mg, 4) != "GGUF") {
        std::cerr << "not GGUF: " << path << "\n";
        return false;
    }
    uint32_t ver = 0;
    f.read((char *) &ver, 4);
    uint64_t n_kv = 0, n_t = 0;
    f.read((char *) &n_t, 8);
    f.read((char *) &n_kv, 8);

    // Sanity caps (huge buffers would be nonsense)
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

    // Tensor table with sanity checks
    std::vector<GRec> recs;
    recs.reserve((size_t) n_t);
    for (uint64_t i = 0; i < n_t; i++)
    {
        std::string name = rd_s(f);
        if (!is_ascii_identifier(name))
        {
            std::cerr << "ERROR: bad tensor name " << name << "\n";
            return false;
        }

        uint32_t nd = 0;
        f.read((char *) &nd, 4);
        if (nd == 0 || nd > 6)
        {
            std::cerr << "ERROR: bad nd=" << nd << " for " << name << "\n";
            return false;
        }

        std::vector<uint64_t> dims(nd);
        for (uint32_t d = 0; d < nd; ++d)
        {
            f.read((char *) &dims[d], 8);
            if (dims[d] == 0 || dims[d] > (uint64_t) 1e10)
            {
                std::cerr << "ERROR: bad dim[" << d << "]=" << dims[d] << " for " << name << "\n";
                return false;
            }
        }

        uint32_t g = 0;
        f.read((char *) &g, 4);
        uint64_t off = 0;
        f.read((char *) &off, 8);
        if (!aligned32(off))
        {
            std::cerr << "ERROR: tensor data offset not 32B aligned for " << name << "\n";
            return false;
        }

        recs.push_back({ std::move(name), nd, std::move(dims), g, off, 0 });
    }

    // Compute sizes by next off / EOF and check monotonicity
    f.seekg(0, std::ios::end);
    size_t fsz = (size_t) f.tellg();
    std::vector<GRec *> ord;
    ord.reserve(recs.size());
    for (auto & r : recs)
        ord.push_back(&r);

    std::sort(ord.begin(), ord.end(), [](auto * a, auto * b) { return a->off < b->off; });
    for (size_t i = 0; i < ord.size(); ++i)
    {
        uint64_t nxt = (i + 1 < ord.size()) ? ord[i + 1]->off : (uint64_t) fsz;
        if (nxt < ord[i]->off) {
            std::cerr << "ERROR: decreasing offsets in baseline\n";
            return false;
        }
        ord[i]->sz = nxt - ord[i]->off;
    }

        // Whole file
    f.seekg(0, std::ios::end);
    std::streampos file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    out.whole.resize(static_cast<size_t>(file_size));               // Allocate exact size

    f.read(reinterpret_cast<char *>(out.whole.data()), file_size);  // Read directly into buffer
    out.recs = std::move(recs);

    return true;
}

// =============== CUDA Dequant Kernels ===============
// Q8_0: per 32 values: [float scale][int8 x 32]
__global__ void k_deq_q8(const int8_t * q, const uint16_t * s, __half * out, int blocks) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= blocks) {
        return;
    }

      __half         h   = __ushort_as_half(s[id]);
    float          d   = __half2float[h];

    if (!isfinite(d))
        d = 0.0f;

    const int8_t * src = q + id * QK;
    __half *       dst = out + id * QK;
#pragma unroll
    for (int i = 0; i < QK; i++) {
        dst[i] = __float2half(d * float(src[i]));
    }
}

namespace q2k_detail {
static constexpr int    QK_K            = 256;
static constexpr size_t BYTES_PER_BLOCK = 84;

__device__ __forceinline__ uint16_t ld_u16_le(const uint8_t * p) {
    // Unaligned little-endian 16-bit load
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
}  // namespace q2k_detail

__global__ void k_deq_q2_k_to_f16(const uint8_t * __restrict__ bytes,
                                  __half * __restrict__ out,
                                  size_t n_elems,
                                  int    blocks) {
    using namespace q2k_detail;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) {
        return;
    }

    const size_t    sb_off = static_cast<size_t>(b) * BYTES_PER_BLOCK;
    const uint8_t * sb     = bytes + sb_off;

    // d and dmin: fp16 (LE) -> float
    const uint16_t d16    = ld_u16_le(sb + 0);
    const uint16_t dmin16 = ld_u16_le(sb + 2);

    float d    = __half2float(__ushort_as_half(d16));
    float dmin = __half2float(__ushort_as_half(dmin16));

    // scales & qs regions
    const uint8_t * scales = sb + 4;   // 16 bytes
    const uint8_t * qs     = sb + 20;  // 64 bytes

    // tail-guard for last partial super-block
    const size_t base        = static_cast<size_t>(b) * QK_K;
    size_t       remaining   = (base >= n_elems) ? 0 : (n_elems - base);
    int          block_elems = (remaining < (size_t) QK_K) ? (int) remaining : QK_K;
    if (block_elems <= 0) {
        return;
    }

// process 16 sub-blocks × 16 weights
#pragma unroll
    for (int sb16 = 0; sb16 < 16; ++sb16) {
        const uint8_t sm    = scales[sb16];
        float         scale = d * float(sm & 0x0F);            // low nibble = scale
        float         minv  = dmin * float((sm >> 4) & 0x0F);  // high nibble = min

        // Match CPU behavior: zero non-finite scale/minv
        if (!isfinite(scale)) {
            scale = 0.0f;
        }
        if (!isfinite(minv)) {
            minv = 0.0f;
        }

        const int sb_base = sb16 * 16;

#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int i = sb_base + j;
            if (i >= block_elems) {
                break;
            }

            // 2-bit unpack: 4 codes per byte
            const int     qi = i;  // 0..255 within super-block
            const uint8_t by = qs[qi >> 2];
            const uint8_t sh = (qi & 3) * 2;
            const uint8_t q  = (by >> sh) & 0x3;  // 0..3

            const float w = scale * float(q) + minv;
            out[base + i] = __float2half_rn(w);  // CPU doesn't post-check w; we cast directly
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
        const uint8_t *       p = bytes;
        for (int b = 0; b < blocks; b++) {
            uint16_t bits;
            memcpy(&bits, p, 2);
            p += 2;
            hs[b] = bits;
            memcpy(&hq[size_t(b) * QK], p, QK);
            p += QK;
        }
        int8_t *   dq  = nullptr;
        uint16_t * ds  = nullptr;
        __half *   out = nullptr;
        CUDA_OK(cudaMalloc(&dq, hq.size()));
        CUDA_OK(cudaMalloc(&ds, hs.size() * sizeof(uint16_t)));
        CUDA_OK(cudaMalloc(&out, n_elems * sizeof(__half)));
        CUDA_OK(cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(ds, hs.data(), hs.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        int th = 256, bl = (blocks + th - 1) / th;
        k_deq_q8<<<bl, th>>>(dq, ds, out, blocks);
        CUDA_OK(cudaDeviceSynchronize());
        cudaFree(dq);
        cudaFree(ds);
        return out;
    }
}
    // Try to dequantize a quant payload by size signature; return device FP16 (caller cudaFree) or nullptr
static __half * dequant_try_Q2_K(const uint8_t * bytes, size_t sz, size_t n_elems)
{
    using namespace q2k_detail;

    if (!bytes || n_elems == 0) {
        return nullptr;
    }

    auto ceil_div = [](size_t a, size_t b) {
        return (a + b - 1) / b;
    };
    const int    blocks = int(ceil_div(n_elems, size_t(QK_K)));
    const size_t expect = size_t(blocks) * BYTES_PER_BLOCK;
    if (sz != expect) {
        // Not a raw Q2_K payload (or wrong size).
        return nullptr;
    }

    // Upload packed payload; allocate output
    uint8_t * d_bytes = nullptr;
    __half *  d_out   = nullptr;

    CUDA_OK(cudaMalloc(&d_bytes, sz));
    CUDA_OK(cudaMemcpy(d_bytes, bytes, sz, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_out, n_elems * sizeof(__half)));

    // Launch 1 thread per super-block
    const int th = 256;
    const int bl = (blocks + th - 1) / th;
    k_deq_q2_k_to_f16<<<bl, th>>>(d_bytes, d_out, n_elems, blocks);
    CUDA_OK(cudaDeviceSynchronize());

    cudaFree(d_bytes);
    return d_out;  // device pointer; free with cudaFree
}

__global__ void k_f32_to_f16(__half * o, const float * x, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        o[i] = __float2half(x[i]);
}

// =============== Main ===============
int main(int argc, char ** argv)
{
    std::string pmp, pfp;
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

    MP mp;
    if (!load_mp(pmp, mp)) {
        std::cerr << "bad mpgguf\n";
        return 1;
    }
    GG gg;
    if (!load_fp(pfp, gg))
    {
        std::cerr << "bad gguf baseline\n";
        return 1;
    }

    // Map baseline by name
    std::unordered_map<std::string, GRec *> truth;
    for (auto & r : gg.recs)
        truth[r.name] = &r;

    auto N = [](const std::vector<uint64_t> & d) -> size_t {
        return numel(d);
    };
    const size_t kMaxElems = size_t(1) << 36;  // Defensive ~64B elements

    // Accumulators
    double s_low = 0, n_low = 0, s_high = 0, n_high = 0, s_hl = 0, n_hl = 0;
    float  m_low = 0, m_high = 0, m_hl = 0;
    size_t validated_high = 0, validated_low = 0;

    auto reduce = [&](const __half * A, const __half * B, size_t n, double & s, double & Nacc, float & m) {
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
        s += ss / 1000000000.0L;
        Nacc += double(n);
        if (mx > m) {
            m = mx;
        }
    };

    for (const auto & r : mp.recs)
    {
        auto it = truth.find(r.name);
        if (it == truth.end())
        {
            if (report)
                std::cerr << "SKIP (missing in baseline): " << r.name << "\n";
            continue;
        }
        const GRec & tr = *it->second;

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
        __half *        d_truth = nullptr;
        const uint8_t * tb      = gg.whole.data() + tr.off;
        if (tr.sz == n * sizeof(__half))
        {
            CUDA_OK(cudaMalloc(&d_truth, tr.sz));
            CUDA_OK(cudaMemcpy(d_truth, tb, tr.sz, cudaMemcpyHostToDevice));
        }
        else if (tr.sz == n * sizeof(float))
        {
            float * df = nullptr;
            CUDA_OK(cudaMalloc(&df, tr.sz));
            CUDA_OK(cudaMemcpy(df, tb, tr.sz, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMalloc(&d_truth, n * sizeof(__half)));
            int th = 256, bl = (int) ((n + th - 1) / th);
            k_f32_to_f16<<<bl, th>>>(d_truth, df, n);
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
        if (r.sz_high && in_range(r.off_high, r.sz_high, mp.data.size()))
        {
            const uint8_t * hb     = mp.data.data() + r.off_high;
            __half *        d_high = dequant_try_Q8_0(hb, (size_t) r.sz_high, n);
            if (d_high)
            {
                reduce(d_high, d_truth, n, s_high, n_high, m_high);
                cudaFree(d_high);
                validated_high++;
            }
            else if (report)
            {
                std::cerr << "WARN: unrecognized HIGH payload for " << r.name << "\n";
            }
        }

        // LOW (legacy scalar 2-bit only; Q2_K/IQ2_* skipped)
        if (r.sz_low && in_range(r.off_low, r.sz_low, mp.data.size()))
        {
            const uint8_t * lb    = mp.data.data() + r.off_low;
            __half *        d_low = dequant_try_Q2_K(lb, (size_t) r.sz_low, n);
            if (d_low)
            {
                reduce(d_low, d_truth, n, s_low, n_low, m_low);
                cudaFree(d_low);
                validated_low++;
            }
            else if (report)
                std::cerr << "NOTE: skipping LOW dequant (likely Q2_K / IQ2_*): " << r.name << "\n";
        }

        cudaFree(d_truth);
    }

    auto outStats = [&](const char * tag, double s, double Nacc, float m, size_t ok) {
        if (Nacc <= 0) {
            std::cout << tag << ": N=0 (no tensors validated)\n";
            return;
        }
        double mse  = s / Nacc;
        double rmse = std::sqrt(mse);
        std::cout << tag << ": tensors=" << ok << "  N=" << (uint64_t) Nacc << "  MSE=" << mse << "  RMSE=" << rmse
                  << "  max =" << m << "\n";
    };

    outStats("HIGH(Q8_0) vs FP16", s_high, n_high, m_high, validated_high);
    outStats("LOW(2-bit)  vs FP16", s_low, n_low, m_low, validated_low);
    if (diffHL)
        outStats("HIGH vs LOW", s_hl, n_hl, m_hl, std::min(validated_high, validated_low));

    return 0;
}
