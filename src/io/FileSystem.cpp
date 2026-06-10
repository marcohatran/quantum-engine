#include "io/FileSystem.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>

#include <arrow/buffer.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>

#include <fmt/format.h>

#include "common/Status.hpp"

namespace quantum::io {

namespace {

bool hasUriScheme(std::string_view s) {
    auto colon = s.find("://");
    return colon != std::string_view::npos && colon > 0;
}

// Hadoop's URI form for a local path omits the authority, producing
// `file:/some/path` (one slash). Arrow's FileSystemFromUri expects the
// standard `file:///some/path` form (three slashes — scheme, empty
// authority, absolute path). Normalize so callers can pass either.
std::string normalizeUri(std::string s) {
    if (s.rfind("file:/", 0) == 0 && s.rfind("file://", 0) != 0) {
        // "file:/x" -> "file:///x"
        return "file:///" + s.substr(6);
    }
    return s;
}

// Arrow requires a global Initialize/Finalize pair for the S3 client. We
// initialize lazily on first S3 use and never finalize — the program exit
// reclaims everything anyway.
// We deliberately do NOT call arrow::fs::FinalizeS3() at exit. Registering
// it via std::atexit races against the AWS SDK's own atexit handlers — by
// the time ours fires, the AWS SDK mutexes are already destroyed, which
// produces a "mutex lock failed" abort on process tear-down. Arrow prints
// a warning when S3 is left uninitialized, but it's harmless at exit and
// preferable to a crash.
void ensureS3Initialized() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        auto status = arrow::fs::InitializeS3(arrow::fs::S3GlobalOptions{});
        if (!status.ok()) {
            raise(ErrorCode::Internal,
                  "FileSystem: InitializeS3 failed: {}", status.ToString());
        }
    });
}

} // namespace

std::pair<std::shared_ptr<arrow::fs::FileSystem>, std::string>
resolveUri(std::string_view uri) {
    std::string uriStr = normalizeUri(std::string(uri));

    // Plain local path with no scheme — wrap as file://.
    if (!hasUriScheme(uriStr)) {
        std::string out_path;
        auto fs_result = arrow::fs::FileSystemFromUri("file://" + uriStr, &out_path);
        if (!fs_result.ok()) {
            raise(ErrorCode::InvalidArgument,
                  "FileSystem: failed to resolve local path '{}': {}",
                  uriStr, fs_result.status().ToString());
        }
        return {fs_result.ValueOrDie(), out_path};
    }

    if (uriStr.starts_with("s3://")) ensureS3Initialized();

    std::string out_path;
    auto fs_result = arrow::fs::FileSystemFromUri(uriStr, &out_path);
    if (!fs_result.ok()) {
        raise(ErrorCode::InvalidArgument,
              "FileSystem: failed to resolve URI '{}': {}",
              uriStr, fs_result.status().ToString());
    }
    return {fs_result.ValueOrDie(), out_path};
}

// ---------------------------------------------------------------------
// Disk cache
// ---------------------------------------------------------------------

namespace {

struct DiskCacheState {
    std::mutex            mu;
    std::string           dir;  // empty = disabled
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
};

DiskCacheState& cacheState() {
    static DiskCacheState s;
    static std::once_flag envInit;
    std::call_once(envInit, [] {
        if (const char* env = std::getenv("QUANTUM_DISK_CACHE_DIR")) {
            s.dir = env;
        }
    });
    return s;
}

// FNV-1a, 64-bit. Cache keys append the URI length and a sanitized tail,
// so even a hash collision needs matching lengths and suffixes.
uint64_t fnv1a(std::string_view s) {
    uint64_t h = 1469598103934665603ULL;
    for (const char c : s) {
        h ^= static_cast<uint8_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

std::string cacheKeyFor(std::string_view uri) {
    std::string tail(uri.substr(uri.size() > 32 ? uri.size() - 32 : 0));
    for (char& c : tail) {
        const bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '_';
        if (!keep) c = '_';
    }
    return fmt::format("{:016x}-{}-{}", fnv1a(uri), uri.size(), tail);
}

// Schemes that resolve in-process or to local disk gain nothing from the
// cache. Everything else (s3 / gs / abfs / …) is treated as remote.
bool isRemoteUri(const std::string& uri) {
    if (!hasUriScheme(uri)) return false;
    if (uri.rfind("file:", 0) == 0) return false;
    if (uri.rfind("mock:", 0) == 0) return false;
    return true;
}

} // namespace

void setDiskCacheDir(std::string dir) {
    auto& st = cacheState();
    std::lock_guard<std::mutex> lock(st.mu);
    st.dir = std::move(dir);
}

DiskCacheStats diskCacheStats() {
    auto& st = cacheState();
    return DiskCacheStats{st.hits.load(), st.misses.load()};
}

void resetDiskCacheStats() {
    auto& st = cacheState();
    st.hits.store(0);
    st.misses.store(0);
}

std::string diskCachePath(std::string_view uri) {
    auto& st = cacheState();
    std::lock_guard<std::mutex> lock(st.mu);
    if (st.dir.empty()) {
        raise(ErrorCode::InvalidArgument, "FileSystem: disk cache not configured");
    }

    std::filesystem::create_directories(st.dir);
    const auto cached =
        std::filesystem::path(st.dir) / cacheKeyFor(std::string(uri));
    if (std::filesystem::exists(cached)) {
        st.hits.fetch_add(1, std::memory_order_relaxed);
        return cached.string();
    }

    // Miss: stream the remote object to a temp file, then rename into
    // place so concurrent processes never observe a partial download.
    auto [fs, path] = resolveUri(uri);
    auto in_r = fs->OpenInputStream(path);
    if (!in_r.ok()) {
        raise(ErrorCode::InvalidArgument,
              "FileSystem: disk cache open('{}') failed: {}",
              std::string(uri), in_r.status().ToString());
    }
    auto in = in_r.ValueOrDie();

    const auto tmp = cached.string() + fmt::format(
        ".tmp{}", std::random_device{}());
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            raise(ErrorCode::Internal,
                  "FileSystem: disk cache can't write '{}'", tmp);
        }
        constexpr int64_t kChunk = 1 << 20;
        while (true) {
            auto buf_r = in->Read(kChunk);
            if (!buf_r.ok()) {
                raise(ErrorCode::Internal,
                      "FileSystem: disk cache read('{}') failed: {}",
                      std::string(uri), buf_r.status().ToString());
            }
            auto buf = buf_r.ValueOrDie();
            if (buf->size() == 0) break;
            out.write(reinterpret_cast<const char*>(buf->data()),
                      static_cast<std::streamsize>(buf->size()));
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, cached, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        if (!std::filesystem::exists(cached)) {
            raise(ErrorCode::Internal,
                  "FileSystem: disk cache rename to '{}' failed", cached.string());
        }
    }
    st.misses.fetch_add(1, std::memory_order_relaxed);
    return cached.string();
}

std::shared_ptr<arrow::io::RandomAccessFile> openReadable(std::string_view uri) {
    {
        auto& st = cacheState();
        bool  enabled;
        {
            std::lock_guard<std::mutex> lock(st.mu);
            enabled = !st.dir.empty();
        }
        if (enabled && isRemoteUri(normalizeUri(std::string(uri)))) {
            const std::string local = diskCachePath(uri);
            auto file_result = arrow::io::ReadableFile::Open(local);
            if (!file_result.ok()) {
                raise(ErrorCode::Internal,
                      "FileSystem: open cached '{}' failed: {}",
                      local, file_result.status().ToString());
            }
            return file_result.ValueOrDie();
        }
    }

    auto [fs, path] = resolveUri(uri);
    auto file_result = fs->OpenInputFile(path);
    if (!file_result.ok()) {
        raise(ErrorCode::InvalidArgument,
              "FileSystem: OpenInputFile('{}') failed: {}",
              std::string(uri), file_result.status().ToString());
    }
    return file_result.ValueOrDie();
}

} // namespace quantum::io
