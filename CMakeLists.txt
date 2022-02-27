cmake_minimum_required(VERSION 3.16)

set(CUDA_TOOLKIT_PATH /usr/local/cuda)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_PATH}/bin/nvcc)

project(MonkeyGL LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CUDA_ARCHITECTURES LESS 60)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
  ${PROJECT_SOURCE_DIR}/cuda_common
  ${CUDA_TOOLKIT_PATH}/include
)

link_directories(
  ${CUDA_TOOLKIT_PATH}/lib64
)

set(SRC_LIST
  ./core/BatchInfo.cpp
  ./core/DataManager.cpp
  ./core/DeviceInfo.cpp
  ./core/Defines.cpp
  ./core/Direction.cpp
  ./core/IRender.cpp
  ./core/Methods.cpp
  ./core/PlaneDefines.cpp
  ./core/PlaneInfo.cpp
  ./core/Point.cpp
  ./core/Render.cpp
  ./core/TransferFunction.cpp
  ./core/VolumeInfo.cpp
  ./core/render.cu
)

add_Library(MonkeyGL SHARED ${SRC_LIST})

set(CUDART_LIBRARY cudart)
set(CUBLASLT_LIBRARY cublasLt)

if(NOT WIN32)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

target_link_libraries(${PROJECT_NAME}
    ${CUDART_LIBRARY}
    ${CUBLASLT_LIBRARY}
)