// mpgguf_build.cpp — streaming build of *.mpgguf (LOW/HIGH) with progress
// Build:  g++ -O3 -std=c++17 -o mpgguf_build mpgguf_build.cpp

#include "parser_util.h"
#include "gguf_parser.h"
#include "mpgguf_parser.h"


using namespace monadd;

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

struct ChunkRef
{
    // we no longer store the data; just a reference to source file
    int      src_id;  // 0=LOW file, 1=HIGH file
    uint64_t src_off;
    uint64_t size;
    uint64_t rel_off;  // destination relative (from DATA start)
};


int main(int argc, char ** argv)
{
    // --high Qwen3-30B-A3B-Q8_0.gguf --low Qwen3-30B-A3B-Q2_K.gguf --out Qwen3-30B-A3B.mpgguf --kv-from high --manifest Qwen3-30B-A3B.mpgguf.manifest.json
    // --high Qwen3-1.7B-Q8_0.gguf --low Qwen3-1.7B-Q2_K.gguf --out Qwen3-1.7B.mpgguf --kv-from high --manifest Qwen3-1.7B.mpgguf.manifest.json
    try
    {
        Args args = parse_args(argc, argv);

        // Parse indices and keep bytes (for dedup probe only)
        std::cout << "Parse GGUF index for high model\n";
        auto idxH = parse_gguf_info(0, args.high);

        std::cout << "Parse GGUF index for low model\n";
        auto idxL = parse_gguf_info(0, args.low);

        // open source files for streaming copy
        std::ifstream fH(args.high, std::ios::binary);
        std::ifstream fL(args.low, std::ios::binary);
        if (!fH || !fL)
            throw std::runtime_error("unable to open inputs");

        // Compute data_sz fields (from offsets)
        auto compute_sizes = [&](GGUFIndex & idx, std::ifstream & f) 
        {
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

        compute_sizes(*idxH, fH);
        compute_sizes(*idxL, fL);

        // Union of tensor names
        std::vector<std::string> names;
        names.reserve(idxH->tensors.size() + idxL->tensors.size());
        for (auto & kv : idxH->tensors)
            names.push_back(kv.name);

        for (auto & kv : idxL->tensors)
        {
            if (std::find_if(idxH->tensors.begin(), idxH->tensors.end(), [&](const auto& item) { return item.name == kv.name;}) == idxH->tensors.end())
                names.push_back(kv.name);
        }
        //std::sort(names.begin(), names.end());

        // KV blob choose
        const std::vector<uint8_t> & kv_blob = (args.kv_from == "high") ? idxH->kv_blob : idxL->kv_blob;

        // Plan: build recs + chunk references (no data yet)
        std::vector<MPRec> recs;
        recs.reserve(names.size());
        std::vector<ChunkRef> chunks;
        chunks.reserve(names.size() * 3);
        uint64_t cur_rel = 0;

        auto append_ref = [&](int src_id, uint64_t src_off, uint64_t size) -> uint64_t
        {
            cur_rel = align_up(cur_rel, 32);
            chunks.push_back({ src_id, src_off, size, cur_rel });
            cur_rel = cur_rel + size;
            return chunks.back().rel_off;
        };

        CTimeMeasure msWrite("Writing to file");

        // Progress
        size_t done = 0, total = names.size();
        std::vector<std::string> exp_temples = { "up_exps", "down_exps", "gate_exps" };
        for (const auto & name : names) {
            const TensorInfo * tH  = nullptr;
            const TensorInfo * tL  = nullptr;
            auto               itH = std::find_if(idxH->tensors.begin(), idxH->tensors.end(), [&](const auto& item) { return item.name == name; });
            if (itH != idxH->tensors.end()) {
                tH = &(*itH);
            }
            auto itL = std::find_if(idxL->tensors.begin(), idxL->tensors.end(), [&](const auto& item) { return item.name == name; });
            if (itL != idxL->tensors.end()) {
                tL = &(*itL);
            }
            const TensorInfo * base = tH ? tH : tL;

            MPRec r;
            r.name = name;
            r.nd   = base->n_dims;
            r.dims = base->dims;

            if (tH && tL)
            {
                if (tL)
                {
                    // we add low precision tensor only for expert sub-layer components
                    if (std::find_if(exp_temples.begin(), exp_temples.end(), [&](const std::string& s) { return name.find(s) != std::string::npos; }) != exp_temples.end())
                    {
                        r.flags |= 1u;
                        r.g_low = tL->ggml_type;
                        r.sz_low = tL->data_sz;
                        r.off_low = append_ref(0, tL->data_off, tL->data_sz);
                    }
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
        monadd::wr_le_u64(out, (uint64_t) kv_blob.size());
        monadd::wr_le_u32(out, (uint32_t) recs.size());
        monadd::wr_le_u32(out, (uint32_t)idxH->kv_cnt);

        // directory
        for (const auto & r : recs)
        {
            monadd::wr_le_u32(out, (uint32_t) r.name.size());
            out.write(r.name.data(), (std::streamsize) r.name.size());
            monadd::wr_le_u32(out, r.nd);
            for (auto d : r.dims)
                monadd::wr_le_u64(out, d);

            monadd::wr_le_u32(out, r.flags);
            monadd::wr_le_u32(out, r.g_low);
            monadd::wr_le_u32(out, r.g_high);
            monadd::wr_le_u32(out, r.g_fp);
            monadd::wr_le_u64(out, r.off_low);
            monadd::wr_le_u64(out, r.sz_low);
            monadd::wr_le_u64(out, r.off_high);
            monadd::wr_le_u64(out, r.sz_high);
            monadd::wr_le_u64(out, r.off_fp);
            monadd::wr_le_u64(out, r.sz_fp);
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
        data_start = align_up(data_start, 32);
        pad_to(data_start);

        // write chunks in order of rel_off
        std::sort(chunks.begin(), chunks.end(),
                  [](const ChunkRef & a, const ChunkRef & b) { return a.rel_off < b.rel_off; });

        std::cout << "[mpgguf] writing data..." << std::endl;
        size_t cdone = 0, ctot = chunks.size();
        for (const auto & ch : chunks)
        {
            uint64_t abs = (uint64_t) data_start + ch.rel_off;
            //pad_to(abs);
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
        std::cout << "[mpgguf] size: " << (out_sz / 1e6) << " MB  (data=" << ((align_up(cur_rel, 32)) / 1e6)
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
                mf << "  ],\n  \"totals\": {\"data_bytes\": " << align_up(cur_rel, 32)
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
