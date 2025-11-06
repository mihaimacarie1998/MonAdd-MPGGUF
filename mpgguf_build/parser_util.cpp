#include "parser_util.h"

namespace monadd
{
    uint64_t align_up(uint64_t off, uint32_t align)
    {
        const uint64_t mask = static_cast<uint64_t>(align) - 1u;
        return (off + mask) & ~mask;
    }

    bool aligned32(uint64_t x)
    {
        return (x & 31ull) == 0ull;
    }

    bool aligned64(uint64_t x)
    {
        return (x & 63ull) == 0ull;
    }

    size_t numel(const std::vector<uint64_t>& d)
    {
        size_t n = 1;
        for (auto v : d) {
            n *= size_t(v);
        }
        return n;
    }

    bool in_range(uint64_t off, uint64_t sz, size_t total)
    {
        return off <= (uint64_t)total && sz <= (uint64_t)total && (off + sz) <= (uint64_t)total;
    }

    uint32_t rd_le_u32(const uint8_t* p)
    {
        uint32_t v;
        memcpy(&v, p, 4);
        return v;
    }

    uint64_t rd_le_u64(const uint8_t* p) {
        uint64_t v;
        memcpy(&v, p, 8);
        return v;
    }

    void wr_le_u32(std::ostream& os, uint32_t v)
    {
        os.write(reinterpret_cast<const char*>(&v), 4);
    }

    void wr_le_u64(std::ostream& os, uint64_t v)
    {
        os.write(reinterpret_cast<const char*>(&v), 8);
    }

    bool is_ascii_identifier(const std::string& s)
    {
        if (s.empty() || s.size() > 512)
            return false;

        for (unsigned char c : s)
        {
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
                c == '/' || c == '-'))
            {
                return false;
            }
        }
        return true;
    }

    std::vector<uint8_t> readStreamData2(std::ifstream& f, size_t sz, size_t offset)
    {
        // Create a vector of the requested size.
        std::vector<uint8_t> buffer(sz);

        f.seekg((std::streampos)offset);

        // Read the data directly into the vector's underlying storage.
        f.read(reinterpret_cast<char*>(buffer.data()), sz);

        // Check how many bytes were actually read.
        size_t bytes_read = f.gcount();

        // Resize the vector to the number of bytes actually read.
        if (bytes_read < sz) {
            buffer.resize(bytes_read);
        }

        return buffer; // Return the vector by value (using move semantics).
    };

    std::vector<uint8_t> readStreamData(std::ifstream& f, size_t sz, size_t offset)
    {
        std::vector<uint8_t> out;

        if (!f.good()) return out;

        // 1) Clear any eof/fail from prior ops; required before seekg on some FUSE filesystems.
        f.clear();

        // 2) Find file size.
        std::istream::pos_type cur = f.tellg();
        f.seekg(0, std::ios::end);
        std::istream::pos_type end = f.tellg();
        if (end <= 0) { f.clear(); return out; }
        const size_t file_size = static_cast<size_t>(end);

        // 3) Validate and clamp request.
        if (offset >= file_size) { f.clear(); return out; }
        const size_t to_read = std::min(sz, file_size - offset);

        // 4) Seek to absolute offset from beginning.
        f.clear(); // (again, in case tellg/setg flipped eof)
        f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!f.good()) { f.clear(); return out; }

        // 5) Read with partial-read handling.
        out.resize(to_read);
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(to_read));
        const auto got = static_cast<size_t>(f.gcount());
        if (got < to_read) {
            out.resize(got);
            // Reset state so later calls can still seek/read.
            f.clear();
        }
        return out;
    }

    void stream_copy(std::ifstream& src, std::ofstream& dst, uint64_t src_off, uint64_t size)
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
}