#include "gguf_parser.h"
#include "parser_util.h"
#include <numeric>


namespace monadd
{
    static const char    GGUF_MAGIC[4] = { 'G', 'G', 'U', 'F' };

    static inline bool is_sane_ggml_type(uint32_t t)
    {
        // ggml types are small enums; keep this permissive but bounded
        return t <= 256;
    }

    static bool overflow_mul_u64(uint64_t a, uint64_t b, uint64_t& out)
    {
        if (b && a > std::numeric_limits<uint64_t>::max() / b)
            return true;

        out = a * b;
        return false;
    }

    // Infer data sizes from offsets and file length, and validate bounds.
    // Works even if tensor offsets are not pre-sorted.
    static bool infer_and_check_sizes(std::vector<TensorInfo>& v, size_t file_len)
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
            auto& t = v[idx[k]];
            if (t.data_off >= file_len)
                return false;

            uint64_t next = (k + 1 < idx.size()) ? v[idx[k + 1]].data_off : (uint64_t)file_len;
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
    static bool try_parse_table_at(Cursor& c,
        uint32_t                  version,
        uint64_t                  n_t,
        size_t                    file_len,
        std::vector<TensorInfo>& out) {
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
    std::shared_ptr<GGUFIndex> parse_gguf_info(const size_t splitId, const std::string& path)
    {
        std::shared_ptr<GGUFIndex> gx = std::make_shared<GGUFIndex>(path);

        auto& f = gx->f;

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
        f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)szThreshold);   // Read directly into buffer

        if (file_len < 4)
            throw std::runtime_error("too small: " + path);

        if (memcmp(buf.data(), GGUF_MAGIC, 4) != 0)
            throw std::runtime_error("not GGUF: " + path);

        Cursor c(buf);
        c.seek(4);

        // ----------- header -----------
        const uint32_t version = c.rd_u32();

        // IMPORTANT: order is n_tensors first, then n_kv
        const uint64_t n_t = c.rd_u64();
        const uint64_t n_kv = c.rd_u64();

        const size_t kv_start = c.tell();
        size_t kv_end = kv_start;

        gx->kv_cnt = n_kv;
        // ----------- normal path: skip KV, then parse tensor table at exact end -----------
        try
        {
            Cursor ck(buf);
            ck.seek(kv_start);
            for (uint64_t i = 0; i < n_kv; ++i)
            {
                (void)ck.rd_string();  // key
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

                gx->kv_blob.assign(buf.begin() + kv_start, buf.begin() + kv_end);
                gx->tensors = tlist;

                uint32_t datablock_start = ct.tell();
                datablock_start = align_up(datablock_start, 32);

                for (auto& t : gx->tensors)
                {
                    t.data_off += datablock_start;
                    t.splitId = splitId;
                }

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
            const size_t probe_to = std::min(file_len, kv_start + 16ull * 1024 * 1024);

            for (size_t pos = probe_from; pos < probe_to; pos += 4) {  // step by 2 for speed/alignment
                Cursor cp(buf);
                cp.seek(pos);

                std::vector<TensorInfo> tlist;
                if (try_parse_table_at(cp, version, n_t, file_len, tlist))
                {
                    if (!infer_and_check_sizes(tlist, file_len))
                        continue;

                    if (pos >= kv_start)
                        gx->kv_blob.assign(buf.begin() + kv_start, buf.begin() + pos);

                    gx->tensors = tlist;
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

                gx->tensors = tlist;

                return gx;
            }
        }

        throw std::runtime_error("unable to locate tensor table in " + path);
    }
}