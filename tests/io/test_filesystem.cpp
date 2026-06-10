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

// Hadoop's URI form for local paths omits the authority ("file:/some/path"
// rather than "file:///some/path"). Delta Lake hands paths to us in this
// shape, so we have to normalize before Arrow's FileSystemFromUri rejects it.
TEST(FileSystem, ResolveHadoopStyleFileUri) {
    auto path = makeTempFile("fs_hadoop_uri_", "hello-hadoop");
    auto [fs, resolved] = io::resolveUri("file:" + path);
    ASSERT_NE(fs, nullptr);
    EXPECT_EQ(fs->type_name(), "local");
    EXPECT_NE(resolved.find("fs_hadoop_uri_"), std::string::npos);
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

// --- Disk cache (Phase 5.44) ---

namespace {

std::string mkTempFile(const std::string& hint, const std::string& contents) {
    auto p = std::filesystem::temp_directory_path() /
             (hint + std::to_string(std::random_device{}()) + ".bin");
    std::ofstream(p, std::ios::binary) << contents;
    return p.string();
}

std::string slurpFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

// RAII: enable the cache for one test, restore disabled state after.
struct CacheGuard {
    explicit CacheGuard(const std::string& dir) {
        io::setDiskCacheDir(dir);
        io::resetDiskCacheStats();
    }
    ~CacheGuard() { io::setDiskCacheDir(""); }
};

} // namespace

TEST(FileSystem, DiskCacheMissThenHit) {
    const std::string payload = "quantum disk cache payload 0123456789";
    const std::string src     = mkTempFile("qcache_src_", payload);
    const auto cacheDir       = (std::filesystem::temp_directory_path() /
        ("qcache_dir_" + std::to_string(std::random_device{}()))).string();
    CacheGuard guard(cacheDir);

    // file:// stands in for an object store — diskCachePath is
    // scheme-agnostic; only openReadable gates on remoteness.
    const std::string uri = "file://" + src;

    const std::string local1 = io::diskCachePath(uri);
    EXPECT_EQ(io::diskCacheStats().misses, 1u);
    EXPECT_EQ(io::diskCacheStats().hits,   0u);
    EXPECT_EQ(slurpFile(local1), payload);

    const std::string local2 = io::diskCachePath(uri);
    EXPECT_EQ(local2, local1);
    EXPECT_EQ(io::diskCacheStats().misses, 1u);
    EXPECT_EQ(io::diskCacheStats().hits,   1u);

    // Hit survives source deletion — reads are served from the cache.
    std::filesystem::remove(src);
    const std::string local3 = io::diskCachePath(uri);
    EXPECT_EQ(slurpFile(local3), payload);
    EXPECT_EQ(io::diskCacheStats().hits, 2u);

    std::filesystem::remove_all(cacheDir);
}

TEST(FileSystem, DiskCacheDistinctUrisDistinctEntries) {
    const std::string a = mkTempFile("qcache_a_", "aaaa");
    const std::string b = mkTempFile("qcache_b_", "bbbb");
    const auto cacheDir = (std::filesystem::temp_directory_path() /
        ("qcache_dir2_" + std::to_string(std::random_device{}()))).string();
    CacheGuard guard(cacheDir);

    const auto pa = io::diskCachePath("file://" + a);
    const auto pb = io::diskCachePath("file://" + b);
    EXPECT_NE(pa, pb);
    EXPECT_EQ(slurpFile(pa), "aaaa");
    EXPECT_EQ(slurpFile(pb), "bbbb");
    EXPECT_EQ(io::diskCacheStats().misses, 2u);

    std::filesystem::remove(a);
    std::filesystem::remove(b);
    std::filesystem::remove_all(cacheDir);
}

TEST(FileSystem, LocalUrisBypassDiskCache) {
    const std::string src = mkTempFile("qcache_local_", "local bytes");
    const auto cacheDir   = (std::filesystem::temp_directory_path() /
        ("qcache_dir3_" + std::to_string(std::random_device{}()))).string();
    CacheGuard guard(cacheDir);

    // openReadable on local paths / file:// must not populate the cache.
    auto f = io::openReadable("file://" + src);
    auto size_r = f->GetSize();
    ASSERT_TRUE(size_r.ok());
    EXPECT_EQ(static_cast<size_t>(size_r.ValueOrDie()), std::string("local bytes").size());
    EXPECT_EQ(io::diskCacheStats().misses, 0u);
    EXPECT_EQ(io::diskCacheStats().hits,   0u);

    std::filesystem::remove(src);
    std::filesystem::remove_all(cacheDir);
}
