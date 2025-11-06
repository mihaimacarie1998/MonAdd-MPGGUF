#pragma once
#include "parser_util.h"

namespace monadd
{
    // ---------- MPGGUF v3 reader ----------

    static const char    MPGG_MAGIC[7] = { 'M', 'P', 'G', 'G', 'U', 'F', '3' };
    static const uint8_t MPGG_VER = 3;

    struct MPRec
    {
        std::string           name;
        uint32_t              nd;
        std::vector<uint64_t> dims;
        uint32_t              flags = 0, g_low = 0, g_high = 0, g_fp = 0;
        uint64_t              off_low = 0, sz_low = 0, off_high = 0, sz_high = 0, off_fp = 0, sz_fp = 0;
    };

    struct MP
    {
        std::vector<MPRec>      recs;
        std::vector<uint8_t>    kv;
        std::ifstream           f;
        size_t                  data_offset;
        size_t                  data_sz;

        bool Open(const std::string& path)
        {
            f.open(path, std::ios::binary);
            if (!f.is_open()) {
                return false;
            }

            return true;
        }
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
                if (mantissa == 0) 
                {
                    // +/- 0.0
                    f = (sign << 31);
                }
                else 
                {
                    // Subnormal half: normalize mantissa
                    uint32_t e = 1; // effective exponent for half subnormals starts at 1
                    while ((mantissa & 0x400u) == 0u) 
                    { // until leading 1 appears at bit 10
                        mantissa <<= 1;
                        --e;
                    }
                    mantissa &= 0x3FFu; // drop the leading 1
                    // Convert exponent bias: (e - 1) + (127 - 15)
                    uint32_t exp32 = static_cast<uint32_t>(static_cast<int32_t>(e) - 1 + 127 - 15);
                    f = (sign << 31) | (exp32 << 23) | (mantissa << 13);
                }
            }
            else if (exp == 0x1Fu) 
            {
                // Inf / NaN
                f = (sign << 31) | 0x7F800000u | (mantissa << 13);
            }
            else 
            {
                // Normalized
                uint32_t exp32 = exp + (127 - 15);
                f = (sign << 31) | (exp32 << 23) | (mantissa << 13);
            }

            float out;
            std::memcpy(&out, &f, sizeof(out));
            return out;
        }
    };

	bool load_mp(const std::string& path, MP& out);
};