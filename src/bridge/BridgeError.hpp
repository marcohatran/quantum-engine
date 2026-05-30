#pragma once

#include <string>

// Shared error-recording surface for the C bridge translation units. Both
// quantum_c.cpp and quantum_arrow.cpp write here; q_last_error() in
// quantum_c.cpp returns the current value.

namespace quantum::bridge {

void        setLastError(std::string msg) noexcept;
const char* lastError() noexcept;

} // namespace quantum::bridge
