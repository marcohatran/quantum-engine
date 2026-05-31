#pragma once

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
std::shared_ptr<arrow::io::RandomAccessFile> openReadable(std::string_view uri);

// Resolve a URI to (FileSystem, path-within-that-FS). Exposed so callers
// that need the filesystem handle directly (e.g. to list, stat) can reuse
// the same dispatching logic.
std::pair<std::shared_ptr<arrow::fs::FileSystem>, std::string>
resolveUri(std::string_view uri);

} // namespace quantum::io
