#pragma once
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
#    define QK 32  // scalar block size used by Q8_0
#endif

namespace monadd
{
    enum GGUFType
    {
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

    enum QuantizationType
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

    struct GRec
    {
        size_t                splitId;
        std::string           name;
        uint32_t              nd;
        std::vector<uint64_t> dims;
        uint32_t              g;
        uint64_t              off, sz;
    };

    struct GG
    {
        std::vector<GRec>    recs;
        std::ifstream f;

        bool Open(const std::string& path)
        {
            f.open(path, std::ios::binary);
            if (!f.is_open()) {
                return false;
            }

            return true;
        }
    };

    struct TensorInfo
    {
        size_t                splitId = 0;
        std::string           name;
        uint32_t              n_dims = 0;
        std::vector<uint64_t> dims;
        uint32_t              ggml_type = 0;
        uint64_t              data_off = 0;
        uint64_t              data_sz = 0;
    };

    struct GGUFIndex
    {
        std::vector<TensorInfo> tensors;
        std::vector<uint8_t>    kv_blob;
        std::ifstream f;

        bool Open(const std::string& path)
        {
            f.open(path, std::ios::binary);
            if (!f.is_open()) {
                std::cerr << "open failed: " << path << "\n";
                return false;
            }

            return true;
        }

        GGUFIndex(const std::string& path) { Open(path); }
        ~GGUFIndex() { f.close(); }
    };

    // ----- mmap-less cursor over a vector -----
    struct Cursor
    {
        const uint8_t* b = nullptr;
        size_t          i = 0, n = 0;
        Cursor() = default;

        Cursor(const std::vector<uint8_t>& buf)
        {
            b = buf.data();
            n = buf.size();
            i = 0;
        }

        static int scalar_size(int t)
        {
            switch (t) {
            case GGUFType::GGUF_U8:
            case GGUFType::GGUF_I8:
            case GGUFType::GGUF_BOOL:
                return 1;
            case GGUFType::GGUF_U16:
            case GGUFType::GGUF_I16:
                return 2;
            case GGUFType::GGUF_U32:
            case GGUFType::GGUF_I32:
            case GGUFType::GGUF_F32:
                return 4;
            case GGUFType::GGUF_U64:
            case GGUFType::GGUF_I64:
            case GGUFType::GGUF_F64:
                return 8;
            default:
                return -1;
            }
        }

        size_t tell() const { return i; }

        void seek(size_t off)
        {
            if (off > n)
                throw std::runtime_error("seek OOB");
            i = off;
        }

        void read_exact(void* out, size_t len)
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
            uint64_t              len = rd_u64();
            // GGUF tensor names / KV keys are small; cap prevents false positives
            static const uint64_t kMaxStr = 4096 * 1024;  // 4 MB is generous
            if (len > kMaxStr)
                throw std::runtime_error("string length over cap");
            if (i + len > n)
                throw std::runtime_error("EOF string");

            std::string s(reinterpret_cast<const char*>(b + i), (size_t)len);
            i += len;
            return s;
        }

        void skip_bytes_blob()
        {
            uint64_t len = rd_u64();
            seek(i + (size_t)len);
        }

        void skip_scalar(int t)
        {
            int s = scalar_size(t);
            if (s < 0)
                throw std::runtime_error("bad scalar");

            seek(i + (size_t)s);
        }

        void skip_value()
        {
            uint32_t t = rd_u32();
            int s = scalar_size((int)t);
            if (s > 0)
            {
                skip_scalar((int)t);
                return;
            }

            if (t == GGUF_STRING)
            {
                (void)rd_string();
                return;
            }

            if (t == GGUF_BYTES)
            {
                skip_bytes_blob();
                return;
            }

            if (t == GGUF_ARRAY)
            {
                uint32_t et = rd_u32();
                uint64_t cnt = rd_u64();
                int      es = scalar_size((int)et);
                if (es > 0)
                {
                    seek(i + (size_t)es * (size_t)cnt);
                    return;
                }

                if (et == GGUF_STRING)
                {
                    for (uint64_t k = 0; k < cnt; k++)
                        (void)rd_string();
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

    std::unique_ptr<GGUFIndex>              parse_gguf_info(const size_t splitId, const std::string& path);

    bool load_fp(const size_t splitId, const std::string& path, GG& out);
};