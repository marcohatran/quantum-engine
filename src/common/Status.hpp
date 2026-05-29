#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/format.h>

namespace quantum {

enum class ErrorCode {
    Ok,
    OutOfMemory,
    InvalidArgument,
    TypeMismatch,
    NotImplemented,
    Internal,
};

class QuantumError : public std::runtime_error {
public:
    QuantumError(ErrorCode code, std::string msg)
        : std::runtime_error(std::move(msg)), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

class QuantumOOM : public QuantumError {
public:
    explicit QuantumOOM(std::string msg)
        : QuantumError(ErrorCode::OutOfMemory, std::move(msg)) {}
};

template <typename... Args>
[[noreturn]] inline void raise(ErrorCode code, fmt::format_string<Args...> fmt, Args&&... args) {
    throw QuantumError(code, fmt::format(fmt, std::forward<Args>(args)...));
}

} // namespace quantum
