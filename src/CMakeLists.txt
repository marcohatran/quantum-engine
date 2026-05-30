add_library(quantum_core STATIC
    common/DataType.cpp
    common/Status.cpp
    memory/Arena.cpp
    memory/MemoryManager.cpp
    memory/Buffer.cpp
    vector/ValidityBitmask.cpp
    vector/PositionList.cpp
    vector/ColumnVector.cpp
    vector/ColumnBatch.cpp
    vector/StringBuilder.cpp
    expression/Literal.cpp
    expression/ColumnRef.cpp
    expression/BinaryExpr.cpp
    expression/kernels/ArithmeticKernels.cpp
    expression/kernels/ComparisonKernels.cpp
    hash/HashTable.cpp
    hash/BytesHashTable.cpp
    hash/KeyCodec.cpp
    operator/ScanOperator.cpp
    operator/FilterOperator.cpp
    operator/ProjectOperator.cpp
    operator/AggregateFunction.cpp
    operator/HashAggregateOperator.cpp
    operator/HashJoinOperator.cpp
)

target_include_directories(quantum_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(quantum_core
    PUBLIC
        fmt::fmt
        absl::base
        absl::span
        absl::strings
        absl::inlined_vector
)

target_compile_features(quantum_core PUBLIC cxx_std_20)

quantum_set_warnings(quantum_core)
quantum_set_simd(quantum_core)
quantum_apply_sanitizers(quantum_core)
