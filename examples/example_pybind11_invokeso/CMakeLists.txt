cmake_minimum_required(VERSION 3.16)

project(example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/../../cuda_common
  ${CUDA_TOOLKIT_PATH}/include
  ${PROJECT_SOURCE_DIR}/../../core
)

add_subdirectory(pybind11)
pybind11_add_module(example example.cpp)

link_directories(
  ${CUDA_TOOLKIT_PATH}/lib64
  # ${PROJECT_SOURCE_DIR}/../../build
)

TARGET_LINK_LIBRARIES(example PRIVATE
  "${PROJECT_SOURCE_DIR}/../../build/libMonkeyGL.so"
)

set_target_properties( example
  PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)