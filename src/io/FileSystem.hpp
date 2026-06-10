#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace arrow {
namespace io { class RandomAccessFile; }
namespace fs { class FileSystem; }
}

namespace quantum::io {

// Open a URI for reading and return a shared Arrow RandomAccessFile.
// Supported schemes (depending on the Arrow build):
//   /local/path           or  file:///local/path  — local filesystem
//   s3://bucket/key                                — Amazon S3
//   gs://bucket/key                                — Google Cloud Storage
//   abfs[s]://...                                  — Azure Blob FS
//   mock:///path                                   — in-process test fs
//
// S3 (and GCS) require a one-time initialization. The helper handles that
// transparently the first time an `s3://` URI is opened on this process.
//
// Disk cache: when a cache directory is configured (setDiskCacheDir, or
// the QUANTUM_DISK_CACHE_DIR environment variable), remote URIs (any
// scheme except file:// and mock://) are downloaded once into the cache
// directory — keyed by a hash of the full URI — and served from local
// disk on every subsequent open. Cache entries are immutable; Delta and
// parquet data files never change in place, so no invalidation is needed
// beyond deleting the cache directory.
std::shared_ptr<arrow::io::RandomAccessFile> openReadable(std::string_view uri);

// Resolve a URI to (FileSystem, path-within-that-FS). Exposed so callers
// that need the filesystem handle directly (e.g. to list, stat) can reuse
// the same dispatching logic.
std::pair<std::shared_ptr<arrow::fs::FileSystem>, std::string>
resolveUri(std::string_view uri);

// Configure the disk-cache directory (created on first use). An empty
// string disables caching. Overrides QUANTUM_DISK_CACHE_DIR.
void setDiskCacheDir(std::string dir);

struct DiskCacheStats {
    uint64_t hits   = 0;
    uint64_t misses = 0;  // downloads
};
DiskCacheStats diskCacheStats();
void           resetDiskCacheStats();

// Run `uri` through the cache regardless of scheme and return the local
// cached file path, downloading on a miss. openReadable routes remote
// schemes through this; exposed so tests can exercise the cache with
// file:// sources standing in for an object store.
std::string diskCachePath(std::string_view uri);

} // namespace quantum::io
