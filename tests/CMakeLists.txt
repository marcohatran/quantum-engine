add_executable(quantum_tests
    common/test_data_type.cpp
    memory/test_buffer.cpp
    memory/test_arena.cpp
    memory/test_memory_manager.cpp
    vector/test_validity_bitmask.cpp
    vector/test_position_list.cpp
    vector/test_column_vector.cpp
    vector/test_column_batch.cpp
    vector/test_string_builder.cpp
    vector/test_dict_string_builder.cpp
    expression/test_literal.cpp
    expression/test_column_ref.cpp
    expression/test_binary_expr.cpp
    expression/test_like_expr.cpp
    operator/test_scan_operator.cpp
    operator/test_filter_operator.cpp
    operator/test_project_operator.cpp
    operator/test_aggregate_function.cpp
    operator/test_hash_aggregate_operator.cpp
    operator/test_hash_join_operator.cpp
    operator/test_string_join.cpp
    operator/test_string_keys.cpp
    operator/test_outer_joins.cpp
    operator/test_dict_e2e.cpp
    hash/test_hash_table.cpp
    e2e/test_pipeline_smoke.cpp
)

target_link_libraries(quantum_tests
    PRIVATE
        quantum_core
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(quantum_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

quantum_set_warnings(quantum_tests)
quantum_apply_sanitizers(quantum_tests)

include(GoogleTest)
gtest_discover_tests(quantum_tests)
