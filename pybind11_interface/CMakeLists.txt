cmake_minimum_required(VERSION 3.16)

project(pyMonkeyGL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/../cuda_common
  ${CUDA_TOOLKIT_PATH}/include
  ${PROJECT_SOURCE_DIR}/../core
)

add_subdirectory(pybind11)
pybind11_add_module(pyMonkeyGL PybindInterface.cpp)

link_directories(
  ${CUDA_TOOLKIT_PATH}/lib64
)

TARGET_LINK_LIBRARIES(pyMonkeyGL PRIVATE
  "${PROJECT_SOURCE_DIR}/../build/libMonkeyGL.so"
)

set_target_properties( pyMonkeyGL
  PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)