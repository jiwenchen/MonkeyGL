cmake_minimum_required(VERSION 3.16)

set(CUDA_TOOLKIT_PATH /usr/local/cuda)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_PATH}/bin/nvcc)

project(example_exe_invokeso LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# set_source_files_properties(add_kernel.cu PROPERTIES LANGUAGE CUDA)

include_directories(
  ${PROJECT_SOURCE_DIR}/../../cuda_common
  ${CUDA_TOOLKIT_PATH}/include
  ${PROJECT_SOURCE_DIR}/../../core
)

link_directories(
  ${CUDA_TOOLKIT_PATH}/lib64
  ${PROJECT_SOURCE_DIR}/../../build
)

set(SRC_LIST example.cpp)

add_Executable(example ${SRC_LIST})

TARGET_LINK_LIBRARIES(example MonkeyGL)