# Setup base
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
set(ENV{BUNSEN_VERSION} 0.0.1)
project(bunsen LANGUAGES CXX VERSION $ENV{BUNSEN_VERSION})

set(BUNSEN_CXX_BUILTINS_DIR bunsen/builtins)
file(REAL_PATH ENV{Torch_DIR} third_party/libtorch/share/cmake/Torch)
set(LIBTORCH_URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)

if(NOT EXISTS third_party/libtorch)
    message(STATUS "Downloading LibTorch...")
    file(DOWNLOAD "${LIBTORCH_URL}" third_party/libtorch.zip SHOW_PROGRESS)

    message(STATUS "Extracting LibTorch...")
    file(ARCHIVE_EXTRACT INPUT third_party/libtorch.zip DESTINATION third_party/)
endif()

if("$ENV{BUNSEN_PRODUCTION}" STREQUAL "1")
    set(BUNSEN_PRODUCTION TRUE)
endif()

if($ENV{DEBUG_BUNSEN})
    set(CMAKE_BUILD_TYPE Debug)
    list(APPEND CMAKE_CXX_FLAGS "-pg")
    message(STATUS "Configuring debug build...")
else()
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Configuring release build...")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/abseil-cpp)

list(APPEND CMAKE_MODULE_PATH third_party)

# Bunsen sources
file(GLOB_RECURSE bunsen_lib_src bunsen/*.h bunsen/*.cpp)
file(GLOB_RECURSE bunsen_utils_src bunsen/builtins/*.cpp)
list(REMOVE_ITEM bunsen_lib_src ${bunsen_utils_src})

file(GLOB_RECURSE bunsen_utils_build_src bunsen/*.h bunsen/*.cpp)
file(REAL_PATH bunsen/PythonBindings.cpp bunsen_py_bindings_src)
list(REMOVE_ITEM bunsen_utils_build_src ${bunsen_utils_src} ${bunsen_py_bindings_src})

include_directories(.)

# Build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY bunsen/py)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

add_custom_target(bunsen_py_lib ${CMAKE_COMMAND} -E env BUNSEN_VERSION=$ENV{BUNSEN_VERSION} pip install .)
add_executable(bunsen_dummy_target ${bunsen_utils_build_src} bunsen/builtins/Dummy.cpp)
target_link_libraries(bunsen_dummy_target PRIVATE
        absl::flat_hash_map
        "${TORCH_PYTHON_LIBRARY}"
        "${TORCH_LIBRARIES}")

add_dependencies(bunsen_dummy_target bunsen_py_lib)

# Build Python library internals
pybind11_add_module(bunsen_internals MODULE ${bunsen_lib_src})
target_compile_definitions(bunsen_internals PRIVATE BUNSEN_PYTHON_MODULE)

if(${BUNSEN_PRODUCTION})
    message(STATUS "Using production mode...")
    target_compile_definitions(bunsen_dummy_target PRIVATE BUNSEN_PRODUCTION)
    target_compile_definitions(bunsen_internals PRIVATE BUNSEN_PRODUCTION)
endif()

target_link_libraries(bunsen_internals PRIVATE
        absl::flat_hash_map
        "${TORCH_PYTHON_LIBRARY}"
        "${TORCH_LIBRARIES}")
