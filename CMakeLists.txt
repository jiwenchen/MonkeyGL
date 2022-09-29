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


set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DFPNG_NO_SSE=0 -msse4.1 -mpclmul")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DFPNG_NO_SSE=0 -msse4.1 -mpclmul")

include_directories(
  ${PROJECT_SOURCE_DIR}/cuda_common
  ${CUDA_TOOLKIT_PATH}/include
  ${PROJECT_SOURCE_DIR}/build/log4cplus-2.0.7/build/install/usr/local/include
  ${PROJECT_SOURCE_DIR}/build/ITK-5.2.1/build/install/include/ITK-5.2
)

link_directories(
  ${CUDA_TOOLKIT_PATH}/lib64
  ${PROJECT_SOURCE_DIR}/build/log4cplus-2.0.7/build/install/usr/local/lib
  ${PROJECT_SOURCE_DIR}/build/ITK-5.2.1/build/lib # Cannot use lib files in ./install/lib, since some files not found while linking. Don't know why.
)

link_libraries(log4cplus)
link_libraries(itkv3p_netlib-5.2)
link_libraries(itkdouble-conversion-5.2)
link_libraries(itkvnl-5.2)
link_libraries(itkvnl_algo-5.2)
link_libraries(itkzlib-5.2)
link_libraries(ITKCommon-5.2)
link_libraries(itksys-5.2)
link_libraries(ITKIOImageBase-5.2)
link_libraries(ITKIONRRD-5.2)
link_libraries(ITKNrrdIO-5.2)
link_libraries(ITKniftiio-5.2)
link_libraries(ITKIONIFTI-5.2)
link_libraries(ITKIOMeta-5.2)

set(SRC_LIST
  ./core/fpng/fpng.cpp
  ./core/Base64.hpp
  ./core/BatchInfo.cpp
  ./core/CPRInfo.cpp
  ./core/CuDataInfo.cpp
  ./core/DataManager.cpp
  ./core/DeviceInfo.cpp
  ./core/Defines.cpp
  ./core/Direction.cpp
  ./core/HelloMonkey.cpp
  ./core/IRender.cpp
  ./core/Logger.cpp
  ./core/MarchingCube.cpp
  ./core/Methods.cpp
  ./core/MPRInfo.cpp
  ./core/ImageReader.cpp
  ./core/ObjectInfo.cpp
  ./core/PlaneInfo.cpp
  ./core/Point.cpp
  ./core/Render.cpp
  ./core/StopWatch.cpp
  ./core/AnnotationUtils.cpp
  ./core/TransferFunction.cpp
  ./core/VolumeInfo.cpp
  ./core/CuDataInfo.cu
  ./core/kernel.cu
  ./core/test.cu
)

add_Library(MonkeyGL SHARED ${SRC_LIST})

SET_TARGET_PROPERTIES(MonkeyGL PROPERTIES VERSION 1.2.1)

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