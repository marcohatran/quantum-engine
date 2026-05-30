cmake_minimum_required(VERSION 3.24)

project(quantum_engine
    VERSION 0.1.0
    DESCRIPTION "Open-core C++ vectorized execution engine for Apache Spark."
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(QUANTUM_BUILD_TESTS       "Build unit tests"                 ON)
option(QUANTUM_BUILD_BENCHMARKS  "Build micro-benchmarks"           OFF)
option(QUANTUM_ENABLE_ASAN       "Enable AddressSanitizer"          OFF)
option(QUANTUM_ENABLE_UBSAN      "Enable UndefinedBehaviorSanitizer" OFF)
option(QUANTUM_ENABLE_TSAN       "Enable ThreadSanitizer"           OFF)
option(QUANTUM_ENABLE_AVX512     "Enable AVX-512 kernels"           OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(CompilerWarnings)
include(Sanitizers)
include(Simd)

find_package(fmt CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

enable_testing()
add_subdirectory(src)

if (QUANTUM_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
    add_subdirectory(tests)
endif()

if (QUANTUM_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)
    add_subdirectory(benchmarks)
endif()
