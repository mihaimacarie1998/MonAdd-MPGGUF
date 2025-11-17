#include "mpgguf_parser.h"

namespace monadd
{
    bool load_mp(const std::string& path, MP& out)
    {

        if (out.Open(path) == false)
            return false;

        auto& f = out.f;
        // Header
        char mg[7];
        f.read(mg, 7);
        uint8_t ver = 0;
        f.read((char*)&ver, 1);
        if (std::string(mg, 7) != MPGG_MAGIC || ver != MPGG_VER)
        {
            std::cerr << "Not MPGGUF3 (" << path << ")\n";
            return false;  // fixed from original snippet
        }

        uint64_t kvsz = 0;
        uint32_t nt = 0, kv_cnt = 0;
        f.read((char*)&kvsz, 8);
        f.read((char*)&nt, 4);
        f.read((char*)&kv_cnt, 4);

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
            f.read((char*)&nl, 4);
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
            f.read((char*)&nd, 4);
            if (nd == 0 || nd > 6)
            {
                std::cerr << "ERROR: mpgguf bad nd=" << nd << " for " << name << "\n";
                return false;
            }
            std::vector<uint64_t> dims(nd);
            for (uint32_t d = 0; d < nd; ++d)
            {
                f.read((char*)&dims[d], 8);
                if (dims[d] == 0 || dims[d] > (uint64_t)1e10)
                {
                    std::cerr << "ERROR: mpgguf bad dim[" << d << "]=" << dims[d] << " for " << name << "\n";
                    return false;
                }
            }

            uint32_t flags = 0, gL = 0, gH = 0, gF = 0;
            f.read((char*)&flags, 4);
            f.read((char*)&gL, 4);
            f.read((char*)&gH, 4);
            f.read((char*)&gF, 4);
            if ((flags & ~0x7u) != 0)
            {
                std::cerr << "ERROR: mpgguf flags reserved bits set for " << name << "\n";
                return false;
            }

            uint64_t oL = 0, sL = 0, oH = 0, sH = 0, oF = 0, sF = 0;
            f.read((char*)&oL, 8);
            f.read((char*)&sL, 8);
            f.read((char*)&oH, 8);
            f.read((char*)&sH, 8);
            f.read((char*)&oF, 8);
            f.read((char*)&sF, 8);

            if ((flags & 0x1) && !aligned64(oL))
            {
                std::cerr << "ERROR: LOW off not 64B aligned\n";
                return false;
            }
            if ((flags & 0x2) && !aligned64(oH))
            {
                std::cerr << "ERROR: HIGH off not 64B aligned\n";
                return false;
            }
            if ((flags & 0x4) && !aligned64(oF))
            {
                std::cerr << "ERROR: FP off not 64B aligned\n";
                return false;
            }

            out.recs.push_back({ std::move(name), nd, std::move(dims), flags, gL, gH, gF, oL, sL, oH, sH, oF, sF });
        }

        // KV
        out.kv.resize(kvsz);
        if (kvsz)
            f.read((char*)out.kv.data(), kvsz);

        // Data region
        // 1. Get the current position
        std::streampos current_pos = f.tellg();
        current_pos = align_up(current_pos, 32);

        // 2. Seek to the end
        f.seekg(0, std::ios::end);

        // 3. Get the end position
        std::streampos end_pos = f.tellg();

        // 4. Calculate remaining size
        std::streamsize remaining_size = end_pos - current_pos;

        // 5. Seek back to the current position
        f.seekg(current_pos);

        out.data_offset = (size_t)current_pos;
        out.data_sz = (size_t)remaining_size;

        return true;
    }
}