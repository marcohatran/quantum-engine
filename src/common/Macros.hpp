#pragma once

#if defined(__GNUC__) || defined(__clang__)
    #define QUANTUM_LIKELY(x)        __builtin_expect(!!(x), 1)
    #define QUANTUM_UNLIKELY(x)      __builtin_expect(!!(x), 0)
    #define QUANTUM_ALWAYS_INLINE    inline __attribute__((always_inline))
    #define QUANTUM_RESTRICT         __restrict__
#elif defined(_MSC_VER)
    #define QUANTUM_LIKELY(x)        (x)
    #define QUANTUM_UNLIKELY(x)      (x)
    #define QUANTUM_ALWAYS_INLINE    __forceinline
    #define QUANTUM_RESTRICT         __restrict
#else
    #define QUANTUM_LIKELY(x)        (x)
    #define QUANTUM_UNLIKELY(x)      (x)
    #define QUANTUM_ALWAYS_INLINE    inline
    #define QUANTUM_RESTRICT
#endif

#define QUANTUM_DISALLOW_COPY(Class)         \
    Class(const Class&)            = delete; \
    Class& operator=(const Class&) = delete

#define QUANTUM_DEFAULT_MOVE(Class)              \
    Class(Class&&) noexcept            = default; \
    Class& operator=(Class&&) noexcept = default
