#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace quantum {

enum class TypeId : uint8_t {
    BOOL,
    INT32,
    INT64,
    FLOAT64,
    STRING,  // reserved for Phase 2 — declared so Schema/serialization can name it
};

struct DataType {
    TypeId id;

    constexpr DataType() noexcept              : id(TypeId::INT64) {}
    constexpr explicit DataType(TypeId t) noexcept : id(t) {}

    constexpr bool operator==(const DataType& other) const noexcept { return id == other.id; }
    constexpr bool operator!=(const DataType& other) const noexcept { return id != other.id; }

    // Returns byte width of one slot in a fixed-width values buffer.
    // Returns -1 for variable-width types (STRING).
    constexpr int32_t fixedWidth() const noexcept {
        switch (id) {
            case TypeId::BOOL:    return 1;  // we store BOOL as one bit, but typed buffers use 1 byte
            case TypeId::INT32:   return 4;
            case TypeId::INT64:   return 8;
            case TypeId::FLOAT64: return 8;
            case TypeId::STRING:  return -1;
        }
        return -1;
    }

    constexpr bool isNumeric() const noexcept {
        return id == TypeId::INT32 || id == TypeId::INT64 || id == TypeId::FLOAT64;
    }

    constexpr bool isFixedWidth() const noexcept { return fixedWidth() > 0; }

    std::string_view name() const noexcept;
    std::string      toString() const;
};

inline constexpr DataType Bool{TypeId::BOOL};
inline constexpr DataType Int32{TypeId::INT32};
inline constexpr DataType Int64{TypeId::INT64};
inline constexpr DataType Float64{TypeId::FLOAT64};

struct Field {
    std::string name;
    DataType    type;
};

class Schema {
public:
    Schema() = default;
    Schema(std::initializer_list<Field> fields) : fields_(fields) {}
    explicit Schema(std::vector<Field> fields) : fields_(std::move(fields)) {}

    size_t            numFields() const noexcept    { return fields_.size(); }
    const Field&      field(size_t i) const noexcept { return fields_[i]; }
    const std::vector<Field>& fields() const noexcept { return fields_; }

private:
    std::vector<Field> fields_;
};

} // namespace quantum
