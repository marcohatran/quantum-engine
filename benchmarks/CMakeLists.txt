add_executable(quantum_benchmarks
    bench_validity_bitmask.cpp
    bench_filter.cpp
    bench_project.cpp
    bench_pipeline.cpp
    bench_hash_aggregate.cpp
)

target_link_libraries(quantum_benchmarks
    PRIVATE
        quantum_core
        benchmark::benchmark
        benchmark::benchmark_main
)

target_include_directories(quantum_benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/src)

quantum_set_warnings(quantum_benchmarks)
