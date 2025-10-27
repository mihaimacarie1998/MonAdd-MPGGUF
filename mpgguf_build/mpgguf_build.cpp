// mpgguf_build.cpp — streaming build of *.mpgguf (LOW/HIGH) with progress
// Build:  g++ -O3 -std=c++17 -o mpgguf_build mpgguf_build.cpp

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <chrono>

class CTimeMeasure
{
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    std::string name;

public:
    // Constructor — starts timing
    explicit CTimeMeasure(const std::string& name = "")
        : start(clock::now()), name(name) {}

    // Destructor — stops timing and prints result
    ~CTimeMeasure()
    {
        auto end = clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (!name.empty())
            std::cout << "[TimeMeasure] " << name << " took "
            << duration.count() / 1000.0 << " ms\n";
        else
            std::cout << "[TimeMeasure] Elapsed: "
            << duration.count() / 1000.0 << " ms\n";
    }
};

static inline uint64_t align64(uint64_t x)
{
    return (x + 63ull) & ~63ull;
}

static inline uint64_t align32(uint64_t x)
{
    return (x + 31ull) & ~31ull;
}

static bool bytes_equal_with_probe(const std::vector<uint8_t>& A,
    size_t                       offA,
    const std::vector<uint8_t>& B,
    size_t                       offB,
    size_t                       n)
{
    if (offA + n > A.size() || offB + n > B.size())
        return false;

    if (n <= 8ull * 1024 * 1024)
        return memcmp(A.data() + offA, B.data() + offB, n) == 0;

    if (memcmp(A.data() + offA, B.data() + offB, 1024 * 1024) != 0)
        return false;

    return memcmp(A.data() + offA + (n - 1024 * 1024), B.data() + offB + (n - 1024 * 1024), 1024 * 1024) == 0;
}


static inline uint32_t rd_le_u32(const uint8_t * p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline uint64_t rd_le_u64(const uint8_t * p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline void wr_le_u32(std::ostream & os, uint32_t v)
{
    os.write(reinterpret_cast<const char *>(&v), 4);
}

static inline void wr_le_u64(std::ostream & os, uint64_t v)
{
    os.write(reinterpret_cast<const char *>(&v), 8);
}

static const char    GGUF_MAGIC[4] = { 'G', 'G', 'U', 'F' };
static const char    MPGG_MAGIC[7] = { 'M', 'P', 'G', 'G', 'U', 'F', '2' };
static const uint8_t MPGG_VER      = 2;

static inline bool is_ascii_identifier(const std::string & s)
{
    if (s.empty() || s.size() > 512)
        return false;

    for (unsigned char c : s)
    {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
              c == '/' || c == '-'))
            return false;
    }
    return true;
}

static inline bool is_sane_ggml_type(uint32_t t)
{
    // ggml types are small enums; keep this permissive but bounded
    return t <= 256;
}

static inline bool aligned32(uint64_t x)
{
    return (x & 31ull) == 0ull;
}

// GGUF value types
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
    switch (t)
    {
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

// ----- mmap-less cursor over a vector -----
struct Cursor
{
    const uint8_t * b = nullptr;
    size_t          i = 0, n = 0;
    Cursor() = default;

    Cursor(const std::vector<uint8_t> & buf)
    {
        b = buf.data();
        n = buf.size();
        i = 0;
    }

    size_t tell() const { return i; }

    void seek(size_t off)
    {
        if (off > n)
            throw std::runtime_error("seek OOB");
        i = off;
    }

    void read_exact(void * out, size_t len)
    {
        if (i + len > n)
            throw std::runtime_error("EOF");
        memcpy(out, b + i, len);
        i += len;
    }

    uint32_t rd_u32()
    {
        uint32_t v;
        read_exact(&v, 4);
        return v;
    }

    uint64_t rd_u64()
    {
        uint64_t v;
        read_exact(&v, 8);
        return v;
    }

    std::string rd_string()
    {
        uint64_t              len     = rd_u64();
        // GGUF tensor names / KV keys are small; cap prevents false positives
        static const uint64_t kMaxStr = 4096 * 1024;  // 4 MB is generous
        if (len > kMaxStr)
            throw std::runtime_error("string length over cap");
        if (i + len > n)
            throw std::runtime_error("EOF string");

        std::string s(reinterpret_cast<const char *>(b + i), (size_t) len);
        i += len;
        return s;
    }

    void skip_bytes_blob()
    {
        uint64_t len = rd_u64();
        seek(i + (size_t) len);
    }

    void skip_scalar(int t)
    {
        int s = scalar_size(t);
        if (s < 0)
            throw std::runtime_error("bad scalar");

        seek(i + (size_t) s);
    }

    void skip_value()
    {
        uint32_t t = rd_u32();
        int s = scalar_size((int) t);
        if (s > 0)
        {
            skip_scalar((int) t);
            return;
        }

        if (t == GGUF_STRING)
        {
            (void) rd_string();
            return;
        }

        if (t == GGUF_BYTES)
        {
            skip_bytes_blob();
            return;
        }

        if (t == GGUF_ARRAY)
        {
            uint32_t et  = rd_u32();
            uint64_t cnt = rd_u64();
            int      es  = scalar_size((int) et);
            if (es > 0)
            {
                seek(i + (size_t) es * (size_t) cnt);
                return;
            }

            if (et == GGUF_STRING)
            {
                for (uint64_t k = 0; k < cnt; k++)
                    (void) rd_string();
                return;
            }

            if (et == GGUF_BYTES)
            {
                for (uint64_t k = 0; k < cnt; k++)
                    skip_bytes_blob();
                return;
            }

            if (et == GGUF_ARRAY)
            {
                for (uint64_t k = 0; k < cnt; k++)
                    skip_value();
                return;
            }

            for (uint64_t k = 0; k < cnt; k++)
                skip_bytes_blob();

            return;
        }
        // unknown -> treat as BYTES
        skip_bytes_blob();
    }
};

struct TensorInfo
{
    std::string           name;
    uint32_t              n_dims = 0;
    std::vector<uint64_t> dims;
    uint32_t              ggml_type = 0;
    uint64_t              data_off  = 0;
    uint64_t              data_sz   = 0;
};

struct GGUFIndex
{
    std::vector<TensorInfo> tensors;
    std::vector<uint8_t>                        kv_blob;
    std::string                                 file_path;
    //std::vector<uint8_t>                        file_bytes;  // keep bytes for dedup probes only
};

static bool overflow_mul_u64(uint64_t a, uint64_t b, uint64_t & out)
{
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        return true;

    out = a * b;
    return false;
}

// Infer data sizes from offsets and file length, and validate bounds.
// Works even if tensor offsets are not pre-sorted.
static bool infer_and_check_sizes(std::vector<TensorInfo> & v, size_t file_len)
{
    if (v.empty())
        return true;

    // Make an index sorted by offset (stable for deterministic behavior)
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return v[a].data_off < v[b].data_off; });

    // Offsets must be within file and unique/non-overlapping
    for (size_t k = 0; k < idx.size(); ++k)
    {
        auto & t = v[idx[k]];
        if (t.data_off >= file_len)
            return false;

        uint64_t next = (k + 1 < idx.size()) ? v[idx[k + 1]].data_off : (uint64_t) file_len;
        if (next <= t.data_off)
            return false;  // overlap or zero-size

        t.data_sz = next - t.data_off;

        if (t.data_off + t.data_sz > file_len)
            return false;
    }
    return true;
}

// ----------------- parsing functions -----------------

// parse a tensor table at current cursor location, version-aware for alignment.
// DOES NOT compute data_sz; that’s done later by infer_and_check_sizes().
static bool try_parse_table_at(Cursor&                    c,
                               uint32_t                  version,
                               uint64_t                  n_t,
                               size_t                    file_len,
                               std::vector<TensorInfo> & out) {
    out.clear();
    try
    {
        for (uint64_t i = 0; i < n_t; ++i)
        {
            TensorInfo t;

            // names in GGUF are UTF-8 strings; allow dots, digits, etc.
            t.name = c.rd_string();
            if (t.name.empty())
                return false;

            t.n_dims = c.rd_u32();
            // GGML usually <=4 dims, but GGUF doesn't enforce; be generous but sane.
            if (t.n_dims == 0 || t.n_dims > 32)
                return false;

            t.dims.resize(t.n_dims);
            uint64_t ne = 1;
            for (uint32_t d = 0; d < t.n_dims; ++d)
            {
                t.dims[d] = c.rd_u64();
                if (t.dims[d] == 0)
                    return false;

                uint64_t tmp;
                if (overflow_mul_u64(ne, t.dims[d], tmp))
                    return false;  // overflow

                ne = tmp;
            }

            t.ggml_type = c.rd_u32();
            if (!is_sane_ggml_type(t.ggml_type))
                return false;

            t.data_off = c.rd_u64();
            if (t.data_off >= file_len)
                return false;

            // Version-aware alignment: v2+ uses 32B alignment for tensor data
            if (version >= 2)
            {
                if (!aligned32(t.data_off))
                    return false;
            }
            // else be lenient for older versions

            out.push_back(std::move(t));
        }
    }
    catch (...)
    {
        return false;
    }

    // Do NOT require strictly increasing offsets here.
    // Sizes and overlap checks are done later in infer_and_check_sizes().

    return true;
}

// Top-level: parse GGUF index (header, KV skip/probe, tensor table), infer sizes.
static GGUFIndex parse_gguf_index(const std::string & path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open: " + path);

    CTimeMeasure ms(path);

    // Get file size
    f.seekg(0, std::ios::end);
    std::streampos file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    // threshold to read the data because all metadata information only will be there.
    size_t szThreshold = 512 * 1024 * 1024;
    const size_t  file_len = static_cast<size_t>(file_size);

    if (szThreshold > file_len)
        szThreshold = file_len;

    std::vector<uint8_t> buf(szThreshold);
    f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) szThreshold);   // Read directly into buffer

    if (file_len < 4)
        throw std::runtime_error("too small: " + path);

    if (memcmp(buf.data(), GGUF_MAGIC, 4) != 0)
        throw std::runtime_error("not GGUF: " + path);

    Cursor c(buf);
    c.seek(4);

    // ----------- header -----------
    const uint32_t version = c.rd_u32();

    // IMPORTANT: order is n_tensors first, then n_kv
    const uint64_t n_t  = c.rd_u64();
    const uint64_t n_kv = c.rd_u64();

    const size_t kv_start = c.tell();
    size_t kv_end   = kv_start;

    // ----------- normal path: skip KV, then parse tensor table at exact end -----------
    try
    {
        Cursor ck(buf);
        ck.seek(kv_start);
        for (uint64_t i = 0; i < n_kv; ++i)
        {
            (void) ck.rd_string();  // key
            ck.skip_value();        // value
        }

        kv_end = ck.tell();

        Cursor ct(buf);
        ct.seek(kv_end);

        std::vector<TensorInfo> tlist;
        if (try_parse_table_at(ct, version, n_t, file_len, tlist))
        {
            // infer sizes and validate
            if (!infer_and_check_sizes(tlist, file_len))
                throw std::runtime_error("tensor size/offset validation failed");

            GGUFIndex gx;
            gx.kv_blob.assign(buf.begin() + kv_start, buf.begin() + kv_end);
            gx.file_path = path;
            //gx.file_bytes.swap(buf);
            gx.tensors = tlist;

            uint32_t datablock_start = ct.tell();
            datablock_start = align32(datablock_start);

            for (auto& t : gx.tensors)
                t.data_off += datablock_start;

            return gx;
        }
    }
    catch (...)
    {
        // fall through to probing paths
    }

    // ----------- probe forward within +16 MiB to find tensor table -----------
    {
        const size_t probe_from = kv_end;
        const size_t probe_to   = std::min(file_len, kv_start + 16ull * 1024 * 1024);

        for (size_t pos = probe_from; pos < probe_to; pos += 4) {  // step by 2 for speed/alignment
            Cursor cp(buf);
            cp.seek(pos);

            std::vector<TensorInfo> tlist;
            if (try_parse_table_at(cp, version, n_t, file_len, tlist))
            {
                if (!infer_and_check_sizes(tlist, file_len))
                    continue;

                GGUFIndex gx;
                if (pos >= kv_start)
                    gx.kv_blob.assign(buf.begin() + kv_start, buf.begin() + pos);

                gx.file_path  = path;
                //gx.file_bytes = buf;  // keep for potential dedup
                gx.tensors = tlist;
                return gx;
            }
        }
    }

    // ----------- last resort: assume no KV, tensor table begins at kv_start -----------
    {
        Cursor ct(buf);
        ct.seek(kv_start);

        std::vector<TensorInfo> tlist;
        if (try_parse_table_at(ct, version, n_t, file_len, tlist))
        {
            if (!infer_and_check_sizes(tlist, file_len))
                throw std::runtime_error("tensor size/offset validation failed (no-KV path)");

            GGUFIndex gx;
            gx.file_path = path;
            //gx.file_bytes.swap(buf);
            gx.tensors = tlist;

            return gx;
        }
    }

    throw std::runtime_error("unable to locate tensor table in " + path);
}


struct Args
{
    std::string high, low, out, kv_from = "high", manifest;
    size_t      log_every = 200;  // tensors
};

static Args parse_args(int argc, char ** argv)
{
    Args a;
    for (int i = 1; i < argc; i++)
    {
        std::string s(argv[i]);
        auto need = [&](const char * flag)
        {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (s == "--high")
            a.high = need("--high");
        else if (s == "--low")
            a.low = need("--low");
        else if (s == "--out")
            a.out = need("--out");
        else if (s == "--kv-from")
            a.kv_from = need("--kv-from");
        else if (s == "--manifest")
            a.manifest = need("--manifest");
        else if (s == "--log-every")
            a.log_every = (size_t) std::stoull(need("--log-every"));
        else
        {
            std::cerr << "Unknown arg: " << s << "\n";
            std::exit(2);
        }
    }
    if (a.high.empty() || a.low.empty() || a.out.empty())
    {
        std::cerr << "Usage: --high <Q8_0.gguf> --low <IQ2_XXS|Q2_K.gguf> --out <model.mpgguf> [--kv-from high|low] "
                     "[--manifest path] [--log-every N]\n";
        std::exit(2);
    }

    if (a.kv_from != "high" && a.kv_from != "low") {
        std::cerr << "--kv-from must be high|low\n";
        std::exit(2);
    }
    return a;
}

struct RecOut
{
    std::string           name;
    uint32_t              nd;
    std::vector<uint64_t> dims;
    uint32_t              flags = 0, g_low = 0, g_high = 0, g_fp = 0;
    uint64_t              off_low = 0, sz_low = 0, off_high = 0, sz_high = 0, off_fp = 0, sz_fp = 0;
};

struct ChunkRef
{
    // we no longer store the data; just a reference to source file
    int      src_id;  // 0=LOW file, 1=HIGH file
    uint64_t src_off;
    uint64_t size;
    uint64_t rel_off;  // destination relative (from DATA start)
};

static void stream_copy(std::ifstream& src, std::ofstream& dst,
    uint64_t src_off, uint64_t size)
{
    // Prepare a buffer exactly 'size' bytes long
    std::vector<char> buf(size);

    // Reset and position
    src.clear();
    dst.clear();
    src.seekg(static_cast<std::streampos>(src_off));

    // Read the requested size
    src.read(buf.data(), static_cast<std::streamsize>(size));
    if (src.gcount() != static_cast<std::streamsize>(size))
        throw std::runtime_error("short read while copying");

    // Write the buffer
    dst.write(buf.data(), static_cast<std::streamsize>(size));
}

int main3(int argc, char ** argv)
{
    // --high Qwen3-30B-A3B-Q8_0.gguf --low Qwen3-30B-A3B-Q2_K.gguf --out Qwen3-30B-A3B.mpgguf --kv-from high --manifest Qwen3-30B-A3B.mpgguf.manifest.json
    // --high Qwen3-1.7B-Q8_0.gguf --low Qwen3-1.7B-Q2_K.gguf --out Qwen3-1.7B.mpgguf --kv-from high --manifest Qwen3-1.7B.mpgguf.manifest.json
    try
    {
        Args args = parse_args(argc, argv);

        // Parse indices and keep bytes (for dedup probe only)
        std::cout << "Parse GGUF index for high model\n";
        GGUFIndex idxH = parse_gguf_index(args.high);

        std::cout << "Parse GGUF index for low model\n";
        GGUFIndex idxL = parse_gguf_index(args.low);

        // open source files for streaming copy
        std::ifstream fH(args.high, std::ios::binary);
        std::ifstream fL(args.low, std::ios::binary);
        if (!fH || !fL)
            throw std::runtime_error("unable to open inputs");

        // Compute data_sz fields (from offsets)
        auto compute_sizes = [&](GGUFIndex & idx, std::ifstream & f) {
            // we only need sizes; get file size
            f.seekg(0, std::ios::end);
            uint64_t fsz = (uint64_t) f.tellg();
            f.seekg(0, std::ios::beg);
            std::vector<TensorInfo *> v;
            v.reserve(idx.tensors.size());
            for (auto & t : idx.tensors)
                v.push_back(&t);

            std::sort(v.begin(), v.end(), [](auto * a, auto * b) { return a->data_off < b->data_off; });
            for (size_t i = 0; i < v.size(); ++i)
            {
                uint64_t nxt  = (i + 1 < v.size()) ? v[i + 1]->data_off : fsz;
                v[i]->data_sz = (nxt > v[i]->data_off) ? (nxt - v[i]->data_off) : 0;
            }
        };

        compute_sizes(idxH, fH);
        compute_sizes(idxL, fL);

        // Union of tensor names
        std::vector<std::string> names;
        names.reserve(idxH.tensors.size() + idxL.tensors.size());
        for (auto & kv : idxH.tensors)
            names.push_back(kv.name);

        for (auto & kv : idxL.tensors)
        {
            if (std::find_if(idxH.tensors.begin(), idxH.tensors.end(), [&](const auto& item) { return item.name == kv.name;}) == idxH.tensors.end())
                names.push_back(kv.name);
        }
        //std::sort(names.begin(), names.end());

        // KV blob choose
        const std::vector<uint8_t> & kv_blob = (args.kv_from == "high") ? idxH.kv_blob : idxL.kv_blob;

        // Plan: build recs + chunk references (no data yet)
        std::vector<RecOut> recs;
        recs.reserve(names.size());
        std::vector<ChunkRef> chunks;
        chunks.reserve(names.size() * 3);
        uint64_t cur_rel = 0;

        auto align64 = [](uint64_t x)
        {
            return (x + 63ull) & ~63ull;
        };

        auto append_ref = [&](int src_id, uint64_t src_off, uint64_t size) -> uint64_t
        {
            cur_rel = align64(cur_rel);
            chunks.push_back({ src_id, src_off, size, cur_rel });
            cur_rel = align64(cur_rel + size);
            return chunks.back().rel_off;
        };

        CTimeMeasure msWrite("Writing to file");

        // Progress
        size_t done = 0, total = names.size();

        for (const auto & name : names) {
            const TensorInfo * tH  = nullptr;
            const TensorInfo * tL  = nullptr;
            auto               itH = std::find_if(idxH.tensors.begin(), idxH.tensors.end(), [&](const auto& item) { return item.name == name; });
            if (itH != idxH.tensors.end()) {
                tH = &(*itH);
            }
            auto itL = std::find_if(idxL.tensors.begin(), idxL.tensors.end(), [&](const auto& item) { return item.name == name; });
            if (itL != idxL.tensors.end()) {
                tL = &(*itL);
            }
            const TensorInfo * base = tH ? tH : tL;

            RecOut r;
            r.name = name;
            r.nd   = base->n_dims;
            r.dims = base->dims;

            if (tH && tL)
            {
                if (tL)
                {
                    r.flags |= 1u;
                    r.g_low   = tL->ggml_type;
                    r.sz_low  = tL->data_sz;
                    r.off_low = append_ref(0, tL->data_off, tL->data_sz);
                }

                if (tH)
                {
                    r.flags |= (1u << 1);
                    r.g_high   = tH->ggml_type;
                    r.sz_high  = tH->data_sz;
                    r.off_high = append_ref(1, tH->data_off, tH->data_sz);
                }
            }
            else
            {
                const TensorInfo * t      = tH ? tH : tL;
                int                src_id = tH ? 1 : 0;
                r.flags |= (1u << 2);
                r.g_fp   = t->ggml_type;
                r.sz_fp  = t->data_sz;
                r.off_fp = append_ref(src_id, t->data_off, t->data_sz);
            }

            recs.emplace_back(std::move(r));

            if (++done % args.log_every == 0 || done == total)
                std::cout << "[mpgguf] planned " << done << "/" << total << " tensors\r" << std::flush;
        }
        std::cout << "\n";

        // ----- Write output (no big pre-allocate) -----
        std::ofstream out(args.out, std::ios::binary);
        if (!out)
            throw std::runtime_error("cannot open output: " + args.out);

        // header
        out.write(MPGG_MAGIC, 7);
        out.put((char) MPGG_VER);
        wr_le_u64(out, (uint64_t) kv_blob.size());
        wr_le_u32(out, (uint32_t) recs.size());

        // directory
        for (const auto & r : recs)
        {
            wr_le_u32(out, (uint32_t) r.name.size());
            out.write(r.name.data(), (std::streamsize) r.name.size());
            wr_le_u32(out, r.nd);
            for (auto d : r.dims)
                wr_le_u64(out, d);

            wr_le_u32(out, r.flags);
            wr_le_u32(out, r.g_low);
            wr_le_u32(out, r.g_high);
            wr_le_u32(out, r.g_fp);
            wr_le_u64(out, r.off_low);
            wr_le_u64(out, r.sz_low);
            wr_le_u64(out, r.off_high);
            wr_le_u64(out, r.sz_high);
            wr_le_u64(out, r.off_fp);
            wr_le_u64(out, r.sz_fp);
        }

        // KV
        if (!kv_blob.empty())
            out.write((const char *) kv_blob.data(), (std::streamsize) kv_blob.size());

        // pad & copy
        auto pad_to = [&](uint64_t abs_off)
            {
                std::streampos cur = out.tellp();
                uint64_t cur_u = (uint64_t)cur;
                if (cur_u < abs_off)
                {
                    static const char zeros[4096] = { 0 };
                    uint64_t left = abs_off - cur_u;
                    while (left > 0)
                    {
                        size_t n = (left > sizeof(zeros)) ? sizeof(zeros) : (size_t)left;
                        out.write(zeros, (std::streamsize)n);
                        left -= n;
                    }
                }
            };

        // data start
        std::streampos data_start = out.tellp();
        data_start = align64(data_start);
        pad_to(data_start);

        // write chunks in order of rel_off
        std::sort(chunks.begin(), chunks.end(),
                  [](const ChunkRef & a, const ChunkRef & b) { return a.rel_off < b.rel_off; });

        std::cout << "[mpgguf] writing data..." << std::endl;
        size_t cdone = 0, ctot = chunks.size();
        for (const auto & ch : chunks)
        {
            uint64_t abs = (uint64_t) data_start + ch.rel_off;
            pad_to(abs);
            if (ch.src_id == 1)
                stream_copy(fH, out, ch.src_off, ch.size);
            else
                stream_copy(fL, out, ch.src_off, ch.size);

            if (++cdone % (args.log_every * 4) == 0 || cdone == ctot)
                std::cout << "[mpgguf] wrote " << cdone << "/" << ctot << " chunks\r" << std::flush;
        }
        std::cout << "\n";

        out.flush();
        out.close();

        // stats
        std::ifstream fo(args.out, std::ios::binary | std::ios::ate);
        auto          out_sz = (uint64_t) fo.tellg();
        std::cout << "[mpgguf] wrote: " << args.out << "\n";
        std::cout << "[mpgguf] size: " << (out_sz / 1e6) << " MB  (data=" << ((align64(cur_rel)) / 1e6)
                  << " MB, kv=" << (kv_blob.size() / 1e3) << " KB)\n";

        // optional manifest (small)
        if (!args.manifest.empty())
        {
            std::ofstream mf(args.manifest);
            if (mf)
            {
                mf << "{\n  \"tensors\": [\n";
                for (size_t i = 0; i < recs.size(); ++i)
                {
                    const auto & r = recs[i];
                    mf << "    {\"name\":\"" << r.name << "\",\"shape\":[";
                    for (size_t k = 0; k < r.dims.size(); ++k)
                    {
                        mf << r.dims[k];
                        if (k + 1 < r.dims.size())
                            mf << ",";
                    }

                    mf << "],\"low_bytes\":" << r.sz_low << ",\"high_bytes\":" << r.sz_high
                       << ",\"fp_bytes\":" << r.sz_fp << "}";
                    if (i + 1 < recs.size())
                        mf << ",";
                    mf << "\n";
                }
                mf << "  ],\n  \"totals\": {\"data_bytes\": " << align64(cur_rel)
                   << ", \"kv_bytes\": " << kv_blob.size() << ", \"file_bytes\": " << out_sz << "}\n}\n";
            }
        }

        return 0;
    }
    catch (const std::exception & e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
