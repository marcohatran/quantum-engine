#include "spill/SpillFile.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "common/Status.hpp"

namespace quantum {

namespace {
std::atomic<uint64_t> g_spillCounter{0};

std::string makeTempPath(std::string_view nameHint) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto seq = g_spillCounter.fetch_add(1, std::memory_order_relaxed);
    auto tmpDir = std::filesystem::temp_directory_path();
    return (tmpDir /
            fmt::format("{}-{}-{}", nameHint, now, seq)).string();
}
} // namespace

SpillFile::SpillFile(size_t stateBytes, std::string nameHint)
    : path_(makeTempPath(nameHint)), stateBytes_(stateBytes) {
    writeFp_ = std::fopen(path_.c_str(), "wb");
    if (!writeFp_) {
        raise(ErrorCode::Internal,
              "SpillFile: failed to open {} for writing (errno={})",
              path_, errno);
    }
}

SpillFile::~SpillFile() {
    if (writeFp_) std::fclose(writeFp_);
    if (readFp_)  std::fclose(readFp_);
    if (!path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
}

SpillFile::SpillFile(SpillFile&& other) noexcept
    : path_(std::move(other.path_)),
      stateBytes_(other.stateBytes_),
      writeFp_(other.writeFp_),
      readFp_(other.readFp_),
      recordCount_(other.recordCount_) {
    other.writeFp_ = nullptr;
    other.readFp_  = nullptr;
    other.path_.clear();
}

SpillFile& SpillFile::operator=(SpillFile&& other) noexcept {
    if (this != &other) {
        if (writeFp_) std::fclose(writeFp_);
        if (readFp_)  std::fclose(readFp_);
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }
        path_        = std::move(other.path_);
        stateBytes_  = other.stateBytes_;
        writeFp_     = other.writeFp_;
        readFp_      = other.readFp_;
        recordCount_ = other.recordCount_;
        other.writeFp_ = nullptr;
        other.readFp_  = nullptr;
        other.path_.clear();
    }
    return *this;
}

void SpillFile::write(const uint8_t* key, int32_t keyLen, const uint8_t* state) {
    if (!writeFp_) raise(ErrorCode::Internal, "SpillFile: write after finishWriting");
    if (keyLen < 0) raise(ErrorCode::InvalidArgument, "SpillFile: keyLen must be >= 0");

    if (std::fwrite(&keyLen, sizeof(keyLen), 1, writeFp_) != 1 ||
        (keyLen > 0 && std::fwrite(key, 1, static_cast<size_t>(keyLen), writeFp_)
                       != static_cast<size_t>(keyLen)) ||
        std::fwrite(state, 1, stateBytes_, writeFp_) != stateBytes_) {
        raise(ErrorCode::Internal, "SpillFile: write failed (errno={})", errno);
    }
    ++recordCount_;
}

void SpillFile::writeNullGroup(const uint8_t* state) {
    if (!writeFp_) raise(ErrorCode::Internal, "SpillFile: write after finishWriting");
    const int32_t sentinel = -1;
    if (std::fwrite(&sentinel, sizeof(sentinel), 1, writeFp_) != 1 ||
        std::fwrite(state, 1, stateBytes_, writeFp_) != stateBytes_) {
        raise(ErrorCode::Internal, "SpillFile: write failed (errno={})", errno);
    }
    ++recordCount_;
}

void SpillFile::finishWriting() {
    if (!writeFp_) return;
    std::fclose(writeFp_);
    writeFp_ = nullptr;
}

SpillFile::Reader::Reader(std::FILE* fp, size_t stateBytes)
    : fp_(fp), stateBytes_(stateBytes), stateBuf_(stateBytes) {}

bool SpillFile::Reader::next() {
    if (!fp_) return false;
    if (std::fread(&keyLen_, sizeof(keyLen_), 1, fp_) != 1) return false;
    if (keyLen_ > 0) {
        keyBuf_.resize(static_cast<size_t>(keyLen_));
        if (std::fread(keyBuf_.data(), 1, keyBuf_.size(), fp_) != keyBuf_.size()) {
            raise(ErrorCode::Internal, "SpillFile reader: short read on key");
        }
    } else {
        keyBuf_.clear();
    }
    if (std::fread(stateBuf_.data(), 1, stateBytes_, fp_) != stateBytes_) {
        raise(ErrorCode::Internal, "SpillFile reader: short read on state");
    }
    return true;
}

SpillFile::Reader SpillFile::reader() {
    finishWriting();
    if (!readFp_) {
        readFp_ = std::fopen(path_.c_str(), "rb");
        if (!readFp_) {
            raise(ErrorCode::Internal,
                  "SpillFile: failed to reopen {} for reading (errno={})",
                  path_, errno);
        }
    }
    std::rewind(readFp_);
    return Reader(readFp_, stateBytes_);
}

} // namespace quantum
