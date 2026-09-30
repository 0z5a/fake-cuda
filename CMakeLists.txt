cmake_minimum_required(VERSION 3.20)
project(fake_cuda VERSION 1.0 LANGUAGES C CXX)

# Build against CUDA 13 headers from this project's uv environment. CUDA 12
# clients use the versioned Driver ABI; neither the CUDA toolkit nor GPU is
# needed to link the shim.
set(CUDA_DRIVER_INCLUDE_DIR
    "${PROJECT_SOURCE_DIR}/.venv-cu130/lib/python3.12/site-packages/nvidia/cu13/include"
    CACHE PATH "CUDA 13 Driver include directory")
if(NOT EXISTS "${CUDA_DRIVER_INCLUDE_DIR}/cuda.h")
    message(FATAL_ERROR "CUDA 13 cuda.h not found at ${CUDA_DRIVER_INCLUDE_DIR}; run bash tests/setup_uv.sh first")
endif()
file(STRINGS "${CUDA_DRIVER_INCLUDE_DIR}/cuda.h" driver_version_line
    REGEX "^#define CUDA_VERSION [0-9]+" LIMIT_COUNT 1)
if(NOT driver_version_line MATCHES "CUDA_VERSION 13[0-9]+")
    message(FATAL_ERROR "CUDA 13 headers required to build this shim: ${driver_version_line}")
endif()
message(STATUS "CUDA Driver headers: ${CUDA_DRIVER_INCLUDE_DIR}")

find_package(Threads REQUIRED)

add_library(fake_cuda_objects OBJECT
    src/driver_api/common.c
    src/driver_api/cu12.c
    src/driver_api/cu13.c
    src/impl/kernel_launch.cpp
    src/impl/perf_model.cpp
    src/impl/kernel_replay.cpp
    src/impl/replay_file.cpp
    src/impl/device_profile.cpp
    src/impl/core.cpp
    src/impl/private.cpp
    src/impl/capture.cpp
    src/impl/graph.cpp
    src/impl/queue_scheduler.cpp
    src/impl/scheduler.cpp
    src/impl/virtual_memory.cpp
    src/impl/virtual_work.cpp)
target_compile_features(fake_cuda_objects PRIVATE c_std_11 cxx_std_23)
target_include_directories(fake_cuda_objects PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_options(fake_cuda_objects PRIVATE -Wall -Wextra)
set_target_properties(fake_cuda_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_EXTENSIONS OFF CXX_EXTENSIONS OFF
    C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
add_library(fake_cuda SHARED $<TARGET_OBJECTS:fake_cuda_objects>)
target_link_libraries(fake_cuda_objects PUBLIC Threads::Threads)
target_link_libraries(fake_cuda PRIVATE Threads::Threads)
set_target_properties(fake_cuda PROPERTIES
    C_EXTENSIONS OFF
    CXX_EXTENSIONS OFF
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME cuda
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    LINK_DEPENDS "${PROJECT_SOURCE_DIR}/exports.map")
target_link_options(fake_cuda PRIVATE
    "LINKER:--version-script=${PROJECT_SOURCE_DIR}/exports.map")

include(CTest)
if(BUILD_TESTING)
    add_executable(replay_probe tests/integration/tooling/replay_probe.cpp)
    target_compile_features(replay_probe PRIVATE cxx_std_23)
    target_link_libraries(replay_probe PRIVATE fake_cuda_objects)
    set_target_properties(replay_probe PROPERTIES CXX_EXTENSIONS OFF)
    foreach(tool IN ITEMS capture_device_profile driver_graph_contract)
        add_executable(${tool} "tests/integration/tooling/${tool}.cpp")
        target_compile_features(${tool} PRIVATE cxx_std_23)
        target_include_directories(${tool} PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
        target_link_libraries(${tool} PRIVATE ${CMAKE_DL_LIBS})
        set_target_properties(${tool} PROPERTIES CXX_EXTENSIONS OFF)
    endforeach()
    foreach(contract IN ITEMS device_profile_contract device_system_contract launch_metadata_contract perf_model_contract replay_file_contract kernel_replay_contract)
        add_executable(${contract} "tests/unit/${contract}.cpp")
        target_compile_features(${contract} PRIVATE cxx_std_23)
        target_link_libraries(${contract} PRIVATE fake_cuda_objects)
        set_target_properties(${contract} PROPERTIES CXX_EXTENSIONS OFF)
    endforeach()
    add_test(NAME device_system_contract COMMAND device_system_contract)
    set_tests_properties(device_system_contract PROPERTIES ENVIRONMENT
        "FAKE_CUDA_SYSTEM=${PROJECT_SOURCE_DIR}/tests/fixtures/directed.system")
    add_test(NAME launch_metadata_contract COMMAND launch_metadata_contract)
    add_test(NAME perf_model_contract COMMAND perf_model_contract)
    add_test(NAME replay_file_contract COMMAND replay_file_contract "${PROJECT_SOURCE_DIR}/tests/fixtures/kernel.timing")
    add_test(NAME kernel_replay_contract COMMAND kernel_replay_contract "${PROJECT_SOURCE_DIR}/tests/fixtures/kernel.timing")
    foreach(count IN ITEMS 1 2 4 8 16 24 32 257)
        add_test(NAME device_count_${count} COMMAND device_profile_contract ${count})
        set_tests_properties(device_count_${count} PROPERTIES ENVIRONMENT "FAKE_CUDA_DEVICE_COUNT=${count}")
    endforeach()
    add_test(NAME device_profile_synthetic COMMAND device_profile_contract 4 synthetic)
    set_tests_properties(device_profile_synthetic PROPERTIES ENVIRONMENT
        "FAKE_CUDA_DEVICE_COUNT=4;FAKE_CUDA_PROFILE=${PROJECT_SOURCE_DIR}/tests/fixtures/synthetic.profile")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME probe_contract COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/unit/probe_contract.py"
            "$<TARGET_FILE:replay_probe>" "${PROJECT_SOURCE_DIR}/tests/integration/tooling/probe_ncu.py")
        add_test(NAME probe_accuracy_contract COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/unit/probe_accuracy_contract.py"
            "$<TARGET_FILE:replay_probe>" "${PROJECT_SOURCE_DIR}/tests/integration/tooling/evaluate_probe.py")
        add_test(NAME device_profile_inputs COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/unit/device_profile_inputs.py"
            "$<TARGET_FILE:device_profile_contract>" "${PROJECT_SOURCE_DIR}/tests/fixtures/synthetic.profile")
    endif()
    foreach(probe IN ITEMS driver_probe virtual_driver_probe virtual_multistream_probe
            multidevice_probe virtual_thread_probe)
        add_executable(${probe} "tests/unit/${probe}.cpp")
        target_compile_features(${probe} PRIVATE cxx_std_23)
        target_include_directories(${probe} PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
        target_link_libraries(${probe} PRIVATE ${CMAKE_DL_LIBS})
        set_target_properties(${probe} PROPERTIES CXX_EXTENSIONS OFF)
    endforeach()
    target_link_libraries(virtual_driver_probe PRIVATE m)
    target_link_libraries(virtual_multistream_probe PRIVATE m)
    target_link_libraries(virtual_thread_probe PRIVATE Threads::Threads)
    add_executable(capture_mode_probe tests/unit/capture_mode_probe.cpp)
    target_compile_features(capture_mode_probe PRIVATE cxx_std_23)
    target_include_directories(capture_mode_probe PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
    target_link_libraries(capture_mode_probe PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    add_test(NAME capture_mode_probe COMMAND capture_mode_probe "$<TARGET_FILE:fake_cuda>")
    add_executable(graph_ordering_probe tests/unit/graph_ordering_probe.cpp)
    target_compile_features(graph_ordering_probe PRIVATE cxx_std_23)
    target_include_directories(graph_ordering_probe PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
    target_link_libraries(graph_ordering_probe PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(capture_mode_probe graph_ordering_probe PROPERTIES CXX_EXTENSIONS OFF)
    foreach(mode IN ITEMS same independent)
        add_test(NAME graph_ordering_${mode}
            COMMAND graph_ordering_probe "$<TARGET_FILE:fake_cuda>" ${mode})
    endforeach()
    add_executable(graph_lifecycle_probe tests/unit/graph_lifecycle_probe.cpp)
    target_compile_features(graph_lifecycle_probe PRIVATE cxx_std_23)
    target_include_directories(graph_lifecycle_probe PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
    target_link_libraries(graph_lifecycle_probe PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(graph_lifecycle_probe PROPERTIES CXX_EXTENSIONS OFF)
    add_test(NAME graph_lifecycle_probe COMMAND graph_lifecycle_probe "$<TARGET_FILE:fake_cuda>")
    set_tests_properties(graph_lifecycle_probe PROPERTIES TIMEOUT 10)

    add_executable(api_contract tests/unit/api_contract.cpp)
    target_compile_features(api_contract PRIVATE cxx_std_23)
    target_include_directories(api_contract PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
    target_link_libraries(api_contract PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(api_contract PROPERTIES CXX_EXTENSIONS OFF)
    add_test(NAME api_contract COMMAND api_contract "$<TARGET_FILE:fake_cuda>")
    set_tests_properties(api_contract PROPERTIES TIMEOUT 20)
    add_test(NAME driver_probe_cuda13 COMMAND driver_probe "$<TARGET_FILE:fake_cuda>")
    add_test(NAME virtual_driver_probe COMMAND virtual_driver_probe "$<TARGET_FILE:fake_cuda>")
    add_test(NAME virtual_multistream_probe COMMAND virtual_multistream_probe "$<TARGET_FILE:fake_cuda>")
    set_tests_properties(virtual_driver_probe virtual_multistream_probe PROPERTIES TIMEOUT 10)

    add_test(NAME multidevice_probe COMMAND multidevice_probe "$<TARGET_FILE:fake_cuda>")
    set_tests_properties(multidevice_probe PROPERTIES
        ENVIRONMENT "FAKE_CUDA_DEVICE_COUNT=2"
        TIMEOUT 15)

    add_test(NAME virtual_thread_probe COMMAND virtual_thread_probe "$<TARGET_FILE:fake_cuda>")
    set_tests_properties(virtual_thread_probe PROPERTIES TIMEOUT 20)

    set(CUDA12_DRIVER_INCLUDE_DIR
        "${PROJECT_SOURCE_DIR}/.venv-cu128/lib/python3.12/site-packages/nvidia/cuda_runtime/include"
        CACHE PATH "Optional CUDA 12 Driver include directory for the ABI probe")
    if(EXISTS "${CUDA12_DRIVER_INCLUDE_DIR}/cuda.h")
        add_executable(driver_probe_cuda12 tests/unit/driver_probe.cpp)
        target_compile_features(driver_probe_cuda12 PRIVATE cxx_std_23)
        target_include_directories(driver_probe_cuda12 PRIVATE "${CUDA12_DRIVER_INCLUDE_DIR}")
        target_link_libraries(driver_probe_cuda12 PRIVATE ${CMAKE_DL_LIBS})
        set_target_properties(driver_probe_cuda12 PROPERTIES CXX_EXTENSIONS OFF)
        add_test(NAME driver_probe_cuda12 COMMAND driver_probe_cuda12 "$<TARGET_FILE:fake_cuda>")
    endif()
endif()
