cmake_minimum_required(VERSION 3.16)

project(example_pybind11)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(pybind11)
pybind11_add_module(example example.cpp)

set_target_properties( example
  PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)