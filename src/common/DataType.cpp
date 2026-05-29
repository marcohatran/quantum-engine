#include "common/DataType.hpp"

namespace quantum {

std::string_view DataType::name() const noexcept {
    switch (id) {
        case TypeId::BOOL:    return "BOOL";
        case TypeId::INT32:   return "INT32";
        case TypeId::INT64:   return "INT64";
        case TypeId::FLOAT64: return "FLOAT64";
        case TypeId::STRING:  return "STRING";
    }
    return "UNKNOWN";
}

std::string DataType::toString() const {
    return std::string(name());
}

} // namespace quantum
