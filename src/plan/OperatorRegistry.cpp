#include "plan/OperatorRegistry.hpp"

#include <utility>

#include "common/Status.hpp"

namespace quantum::plan {

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    return registry;
}

void OperatorRegistry::registerFactory(std::string     extensionType,
                                       OperatorFactory factory) {
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] =
        factories_.emplace(std::move(extensionType), std::move(factory));
    if (!inserted) {
        raise(ErrorCode::InvalidArgument,
              "OperatorRegistry: extension type '{}' already registered",
              it->first);
    }
}

bool OperatorRegistry::contains(const std::string& extensionType) const {
    std::lock_guard<std::mutex> lock(mu_);
    return factories_.find(extensionType) != factories_.end();
}

OperatorPtr OperatorRegistry::build(const std::string& extensionType,
                                    ExtensionContext   ctx) const {
    OperatorFactory factory;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = factories_.find(extensionType);
        if (it == factories_.end()) {
            raise(ErrorCode::NotImplemented,
                  "OperatorRegistry: no factory registered for extension "
                  "type '{}'", extensionType);
        }
        factory = it->second;
    }
    return factory(std::move(ctx));
}

void OperatorRegistry::clearForTesting() {
    std::lock_guard<std::mutex> lock(mu_);
    factories_.clear();
}

} // namespace quantum::plan
