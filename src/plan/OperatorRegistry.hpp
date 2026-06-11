#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "operator/Operator.hpp"

namespace quantum {

class MemoryManager;

namespace plan {

// Everything a plugin factory needs to construct its operator: the
// already-built child subtrees (wrapped in MetricsOperators when metrics
// are enabled), the engine's MemoryManager, and the plan fragment payload.
// The payload is opaque to the engine — by convention a serialized
// Substrait relation the plugin decodes itself, but any encoding works.
struct ExtensionContext {
    MemoryManager*           mm = nullptr;
    std::vector<OperatorPtr> children;
    std::string              payload;
};

using OperatorFactory = std::function<OperatorPtr(ExtensionContext)>;

// Process-global registry mapping an extension type (e.g.
// "vendor.geo_join") to the factory that builds its native operator.
// This is the open-core seam: plugins register at startup and plans
// reference them via ExtensionNode without the engine knowing the
// operator. Thread-safe registration/lookup; factories themselves run on
// the single-threaded build path.
class OperatorRegistry {
public:
    static OperatorRegistry& instance();

    // Raises InvalidArgument on duplicate registration.
    void registerFactory(std::string extensionType, OperatorFactory factory);

    bool contains(const std::string& extensionType) const;

    // Builds via the registered factory; raises NotImplemented when the
    // extension type is unknown.
    OperatorPtr build(const std::string& extensionType, ExtensionContext ctx) const;

    void clearForTesting();

private:
    OperatorRegistry() = default;

    mutable std::mutex                               mu_;
    std::unordered_map<std::string, OperatorFactory> factories_;
};

} // namespace plan
} // namespace quantum
