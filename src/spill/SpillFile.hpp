#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "common/Macros.hpp"

namespace quantum {

// Simple binary on-disk format for spilled HashAggregate partial results.
// Each record is:
//   [int32 keyLen]                 // -1 sentinel = "null group" record
//   [keyLen bytes of key]          // empty when keyLen == -1
//   [int32 stateLen]               // varies per record for out-of-line
//   [stateLen bytes of state]      //   aggregate state (strings, sets)
//
// SpillFile owns a temp file on disk created in the system temp directory.
// On destruction the file is unlinked. No compression, no checksums, no
// cross-process sharing.
class SpillFile {
public:
    explicit SpillFile(size_t stateBytes, std::string nameHint = "quantum_spill");
    ~SpillFile();

    QUANTUM_DISALLOW_COPY(SpillFile);
    SpillFile(SpillFile&&) noexcept;
    SpillFile& operator=(SpillFile&&) noexcept;

    // 3-arg forms write `stateBytes` (the ctor arg) bytes of state.
    void write(const uint8_t* key, int32_t keyLen, const uint8_t* state);
    void write(const uint8_t* key, int32_t keyLen,
               const uint8_t* state, size_t stateLen);
    void writeNullGroup(const uint8_t* state);
    void writeNullGroup(const uint8_t* state, size_t stateLen);

    // Closes the write handle and prepares for reading.
    void finishWriting();

    class Reader {
    public:
        bool        next();
        bool        isNullGroup() const noexcept { return keyLen_ == -1; }
        const uint8_t* key()      const noexcept { return keyBuf_.data(); }
        size_t      keyLen()      const noexcept { return static_cast<size_t>(keyLen_); }
        const uint8_t* state()    const noexcept { return stateBuf_.data(); }
        size_t      stateLen()    const noexcept { return stateBuf_.size(); }

    private:
        friend class SpillFile;
        Reader(std::FILE* fp, size_t stateBytes);

        std::FILE*           fp_;
        int32_t              keyLen_ = 0;
        std::vector<uint8_t> keyBuf_;
        std::vector<uint8_t> stateBuf_;
    };

    Reader reader();

    const std::string& path()       const noexcept { return path_; }
    size_t             recordCount() const noexcept { return recordCount_; }

private:
    std::string  path_;
    size_t       stateBytes_;
    std::FILE*   writeFp_ = nullptr;
    std::FILE*   readFp_  = nullptr;
    size_t       recordCount_ = 0;
};

} // namespace quantum
