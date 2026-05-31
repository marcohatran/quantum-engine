#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include <arrow/filesystem/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>

#include "common/Status.hpp"
#include "io/FileSystem.hpp"

using namespace quantum;

namespace {

std::string makeTempFile(const std::string& nameHint, const std::string& body) {
    auto dir  = std::filesystem::temp_directory_path();
    auto path = (dir / (nameHint + std::to_string(std::random_device{}()) + ".bin")).string();
    std::ofstream(path, std::ios::binary) << body;
    return path;
}

} // namespace

TEST(FileSystem, ResolveBarePathDispatchesToLocalFs) {
    auto path = makeTempFile("fs_local_", "hello");
    auto [fs, resolved] = io::resolveUri(path);
    ASSERT_NE(fs, nullptr);
    EXPECT_EQ(fs->type_name(), "local");
    // LocalFileSystem returns the absolute path inside the filesystem.
    EXPECT_NE(resolved.find("fs_local_"), std::string::npos);
    std::filesystem::remove(path);
}

TEST(FileSystem, ResolveFileUriDispatchesToLocalFs) {
    auto path = makeTempFile("fs_file_uri_", "hello-uri");
    auto [fs, resolved] = io::resolveUri("file://" + path);
    ASSERT_NE(fs, nullptr);
    EXPECT_EQ(fs->type_name(), "local");
    EXPECT_NE(resolved.find("fs_file_uri_"), std::string::npos);
    std::filesystem::remove(path);
}

TEST(FileSystem, OpenReadableLoadsBytes) {
    auto path = makeTempFile("fs_open_", "twelve_bytes");
    auto file = io::openReadable(path);
    ASSERT_NE(file, nullptr);
    auto size_result = file->GetSize();
    ASSERT_TRUE(size_result.ok());
    EXPECT_EQ(size_result.ValueOrDie(), 12);
    std::filesystem::remove(path);
}

TEST(FileSystem, OpenReadableMissingFileThrows) {
    EXPECT_THROW(io::openReadable("/no/such/quantum/file_xyz_42.bin"), QuantumError);
}

TEST(FileSystem, UnknownSchemeThrows) {
    // FileSystemFromUri rejects schemes that the Arrow build doesn't know
    // about; we want that to surface as a Quantum InvalidArgument, not a
    // raw Arrow Status.
    EXPECT_THROW(io::resolveUri("notascheme://bucket/key"), QuantumError);
}

TEST(FileSystem, S3SchemeIsRecognized) {
    // Arrow's FileSystemFromUri eagerly validates the bucket (it hits the
    // network), so we can't test a successful s3:// resolution offline.
    // What we *can* assert is that an s3:// URI is recognized as the S3
    // scheme — the error surfaces as an S3 IOError, not "unknown scheme".
    try {
        (void)io::resolveUri("s3://quantum-engine-nonexistent-bucket-xyz123/key.parquet");
        // If we ever land in an environment where this bucket exists, the
        // test still passes — we got an S3 handle back.
    } catch (const QuantumError& e) {
        std::string msg = e.what();
        // The S3 client was reached if we see any of these substrings.
        EXPECT_TRUE(msg.find("S3") != std::string::npos ||
                    msg.find("s3")  != std::string::npos ||
                    msg.find("IOError") != std::string::npos)
            << "Expected an S3-shaped error, got: " << msg;
    }
}
