# Generate protobuf C++ sources for the plan IR.
set(QUANTUM_PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/plan/proto/quantum_plan.proto)
protobuf_generate_cpp(QUANTUM_PROTO_SRCS QUANTUM_PROTO_HDRS ${QUANTUM_PROTO_FILES})

add_library(quantum_core SHARED
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
    vector/DictStringBuilder.cpp
    expression/Literal.cpp
    expression/ColumnRef.cpp
    expression/BinaryExpr.cpp
    expression/LikeExpr.cpp
    expression/kernels/ArithmeticKernels.cpp
    expression/kernels/ComparisonKernels.cpp
    hash/HashTable.cpp
    hash/BytesHashTable.cpp
    hash/VarBytesHashTable.cpp
    hash/KeyCodec.cpp
    operator/ScanOperator.cpp
    operator/FilterOperator.cpp
    operator/ProjectOperator.cpp
    operator/AggregateFunction.cpp
    operator/HashAggregateOperator.cpp
    operator/HashJoinOperator.cpp
    spill/SpillFile.cpp
    io/CsvReader.cpp
    io/DeltaTableReader.cpp
    io/FileSystem.cpp
    io/ParquetReader.cpp
    plan/Plan.cpp
    plan/PlanBuilder.cpp
    plan/PlanJson.cpp
    plan/PlanProto.cpp
    bridge/quantum_c.cpp
    bridge/quantum_arrow.cpp
    ${QUANTUM_PROTO_SRCS}
)

target_include_directories(quantum_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}     # for the generated *.pb.h
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
        nlohmann_json::nlohmann_json
        protobuf::libprotobuf
        Arrow::arrow_shared
        Parquet::parquet_shared
)

target_compile_features(quantum_core PUBLIC cxx_std_20)

quantum_set_warnings(quantum_core)
quantum_set_simd(quantum_core)
quantum_apply_sanitizers(quantum_core)

# Suppress conversion / sign-conversion warnings inside the generated .pb.cc
# files — they intentionally narrow types for the wire format.
set_source_files_properties(${QUANTUM_PROTO_SRCS} PROPERTIES
    COMPILE_FLAGS "-Wno-conversion -Wno-sign-conversion -Wno-shadow")
