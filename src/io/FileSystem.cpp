#include "io/FileSystem.hpp"

#include <atomic>
#include <mutex>
#include <string>

#include <arrow/filesystem/api.h>
#include <arrow/io/file.h>
#include <arrow/io/interfaces.h>

#include "common/Status.hpp"

namespace quantum::io {

namespace {

bool hasUriScheme(std::string_view s) {
    auto colon = s.find("://");
    return colon != std::string_view::npos && colon > 0;
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
    std::string uriStr(uri);

    // Plain local path with no scheme — wrap as file://.
    if (!hasUriScheme(uri)) {
        std::string out_path;
        auto fs_result = arrow::fs::FileSystemFromUri("file://" + uriStr, &out_path);
        if (!fs_result.ok()) {
            raise(ErrorCode::InvalidArgument,
                  "FileSystem: failed to resolve local path '{}': {}",
                  uriStr, fs_result.status().ToString());
        }
        return {fs_result.ValueOrDie(), out_path};
    }

    if (uri.starts_with("s3://")) ensureS3Initialized();

    std::string out_path;
    auto fs_result = arrow::fs::FileSystemFromUri(uriStr, &out_path);
    if (!fs_result.ok()) {
        raise(ErrorCode::InvalidArgument,
              "FileSystem: failed to resolve URI '{}': {}",
              uriStr, fs_result.status().ToString());
    }
    return {fs_result.ValueOrDie(), out_path};
}

std::shared_ptr<arrow::io::RandomAccessFile> openReadable(std::string_view uri) {
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
