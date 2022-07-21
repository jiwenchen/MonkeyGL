#! /bin/bash

source_path=$(pwd)

build_path=${source_path}"/build"

py_source_path=${source_path}"/pybind11_interface" 
py_build_path=${source_path}"/pybind11_interface/build"

build_type=$2

makesure_folder(){
  if [ ! -d "$1" ];then
    mkdir "$1"
  fi
}

build_zlib() {
    makesure_folder "${build_path}"
    cd "${build_path}" || exit
    if [ "${build_type}" == "Clean" ]; then
      printf "clean zlib build"
      rm -rf ./zlib-1.2.11
    else
      if [ ! -d "./zlib-1.2.11" ];then
        tar -xvzf ../ThirdPartyDownloads/zlib-1.2.11.tar.gz
      fi
      cd ./zlib-1.2.11 || exit

      if [ ! -d "./build" ];then
        mkdir build
        cd ./build || exit
        cmake ../ -DCMAKE_BUILD_TYPE="${build_type}"
        make DESTDIR=./install install
      fi
    fi
}

build_webp() {
    makesure_folder "${build_path}"
    cd "${build_path}" || exit
    if [ "${build_type}" == "Clean" ]; then
      printf "clean webp build"
      rm -rf ./libwebp-1.2.2
    else
      if [ ! -d "./libwebp-1.2.2" ];then
        tar -xvzf ../ThirdPartyDownloads/libwebp-1.2.2.tar.gz
      fi
      cd ./libwebp-1.2.2 || exit

      if [ ! -d "./build" ];then
        mkdir build
        cd ./build || exit
        cmake ../ -DCMAKE_BUILD_TYPE="${build_type}" -DBUILD_SHARED_LIBS=true
        make DESTDIR=./install install
      fi
    fi
}

build_log_lib() {
    makesure_folder "${build_path}"
    cd "${build_path}" || exit
    if [ "${build_type}" == "Clean" ]; then
      printf "clean log build"
      rm -rf ./log4cplus-2.0.7
    else
      if [ ! -d "./log4cplus-2.0.7" ];then
        unzip ../ThirdPartyDownloads/log4cplus-2.0.7.zip
      fi
      cd ./log4cplus-2.0.7 || exit

      if [ ! -d "./build" ];then
        mkdir build
        cd ./build || exit
        cmake ../ -DCMAKE_BUILD_TYPE="${build_type}"
        make DESTDIR=./install install
      fi
    fi
}

build_itk_lib() {
    makesure_folder ${build_path}
    cd ${build_path}
    if [ ${build_type} == "Clean" ]; then
      echo "clean itk build"
      rm -rf ./ITK-5.2.1
    else
      if [ ! -d "./ITK-5.2.1" ];then
        unzip ../ThirdPartyDownloads/ITK-5.2.1.zip
      fi
      cd ./ITK-5.2.1

      if [ ! -d "./build" ];then
        mkdir build
        cd ./build
        cmake ../ -DCMAKE_BUILD_TYPE:STRING=${build_type} -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=true -DCMAKE_INSTALL_PREFIX="./install"
        mkdir install
        make -j 8 install ./install
      fi
    fi
}

build_cpp_lib() {
    makesure_folder "${build_path}"
    cd "${build_path}" || exit
    if [ "${build_type}" == "Clean" ]; then
      printf "clean cpp build"
      rm -rf ./*
    else
      cmake ../ -DCMAKE_BUILD_TYPE="${build_type}" -DSSE=1
      make
    fi
}

build_cpp() {
    build_log_lib
    build_itk_lib
    build_cpp_lib
}

build_pybind() {
    cd "${py_source_path}" || exit
    if [ ! -d "./pybind11" ];then
      tar -xvzf ../ThirdPartyDownloads/pybind11.tar.gz ./
    fi
    makesure_folder "${py_build_path}"
    cd "${py_build_path}" || exit
    if [ "${build_type}" == "Clean" ]; then
      printf "clean pybind build"
      rm -rf ./*
    else
      cmake ../ -DCMAKE_BUILD_TYPE="${build_type}"
      make
    fi
}

if [ "$1" == "cpp" ]; then
    build_cpp
elif [ "$1" == "itk" ]; then
    build_itk_lib
elif [ "$1" == "log" ]; then
    build_log_lib
elif [ "$1" == "pybind" ]; then
    build_pybind
elif [ "$1" == "all" ]; then
    build_cpp
    build_pybind
else
    printf "invalid project"
fi