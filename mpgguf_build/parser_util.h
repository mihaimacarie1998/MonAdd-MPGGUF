#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

namespace monadd
{
    uint64_t                    align_up(uint64_t off, uint32_t align);

    bool                        aligned32(uint64_t x);

    bool                        aligned64(uint64_t x);

    size_t                      numel(const std::vector<uint64_t>& d);

    bool                        in_range(uint64_t off, uint64_t sz, size_t total);

    uint32_t                    rd_le_u32(const uint8_t* p);

    uint64_t                    rd_le_u64(const uint8_t* p);

    void                        wr_le_u32(std::ostream& os, uint32_t v);

    void                        wr_le_u64(std::ostream& os, uint64_t v);

    bool                        is_ascii_identifier(const std::string& s);

    std::vector<uint8_t>        readStreamData(std::ifstream& f, size_t sz, size_t offset);

    std::vector<uint8_t>        readStreamData2(std::ifstream& f, size_t sz, size_t offset);

    void                        stream_copy(std::ifstream& src, std::ofstream& dst, uint64_t src_off, uint64_t size);

    class CTimeMeasure
    {
        using clock = std::chrono::high_resolution_clock;
        clock::time_point start;
        std::string name;

    public:
        // Constructor — starts timing
        explicit CTimeMeasure(const std::string& name = "")
            : start(clock::now()), name(name) {
        }

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

}