argNum=$#
source_path=$(pwd)

build_path=${source_path}"/build"
install_path=${source_path}"/install"

py_source_path=${source_path}"/pybind11_interface" 
py_build_path=${source_path}"/pybind11_interface/build"

build_type=$2

makesure_folder(){
  if [ ! -d $1 ];then
    mkdir $1
  fi
}

remove_folder(){
  if [ -d $1 ];then
    echo "remove folder: " $1
    rm -rf $1
  fi
}

build_zlib() {
    makesure_folder ${build_path}
    cd ${build_path}
    if [ ${build_type} == "Clean" ]; then
      echo "clean zlib build"
      rm -rf ./zlib-1.2.11
    else
      if [ ! -d "./zlib-1.2.11" ];then
        tar -xvzf ../ThirdPartyDownloads/zlib-1.2.11.tar.gz
      fi
      cd ./zlib-1.2.11

      if [ ! -d "./build" ];then
        mkdir build
        cd ./build
        cmake ../ -DCMAKE_BUILD_TYPE=${build_type}
        make DESTDIR=./install install
      fi
    fi
}

build_webp() {
    makesure_folder ${build_path}
    cd ${build_path}
    if [ ${build_type} == "Clean" ]; then
      echo "clean webp build"
      rm -rf ./libwebp-1.2.2
    else
      if [ ! -d "./libwebp-1.2.2" ];then
        tar -xvzf ../ThirdPartyDownloads/libwebp-1.2.2.tar.gz
      fi
      cd ./libwebp-1.2.2

      if [ ! -d "./build" ];then
        mkdir build
        cd ./build
        cmake ../ -DCMAKE_BUILD_TYPE=${build_type} -DBUILD_SHARED_LIBS=true
        make DESTDIR=./install install
      fi
    fi
}

build_log_lib() {
    makesure_folder ${build_path}
    cd ${build_path}
    if [ ${build_type} == "Clean" ]; then
      echo "clean log build"
      rm -rf ./log4cplus-2.0.7
    else
      if [ ! -d "./log4cplus-2.0.7" ];then
        unzip ../ThirdPartyDownloads/log4cplus-2.0.7.zip
      fi
      cd ./log4cplus-2.0.7

      if [ ! -d "./build" ];then
        mkdir build
        cd ./build
        cmake ../ -DCMAKE_BUILD_TYPE=${build_type}
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
    makesure_folder ${build_path}
    cd ${build_path}
    if [ ${build_type} == "Clean" ]; then
      echo "clean cpp build"
      rm -rf ./*
    else
      cmake ../ -DCMAKE_BUILD_TYPE=${build_type} -DSSE=1
      make
    fi
}

build_cpp() {
    build_log_lib
    build_itk_lib
    build_cpp_lib
}

build_pybind() {
    cd ${py_source_path}
    if [ ! -d "./pybind11" ];then
      tar -xvzf ../ThirdPartyDownloads/pybind11.tar.gz ./
    fi
    makesure_folder ${py_build_path}
    cd ${py_build_path}
    if [ ${build_type} == "Clean" ]; then
      echo "clean pybind build"
      rm -rf ./*
    else
      cmake ../ -DCMAKE_BUILD_TYPE=${build_type}
      make
    fi
}

install() {
    makesure_folder ${install_path}
    cd ${install_path}
    makesure_folder ${install_path}/"lib"
    cd ${install_path}/"lib"
    rm -rf ./*
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKCommon-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libitkdouble-conversion-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKIOImageBase-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKIONIFTI-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKIONRRD-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKMetaIO-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKniftiio-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKNrrdIO-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libitksys-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libitkv3p_netlib-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libitkzlib-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libitkvnl_algo-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKznz-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libitkvnl-5.2.so.1" ./
    cp ${build_path}"/ITK-5.2.1/build/lib/libITKIOMeta-5.2.so.1" ./
    cp ${build_path}"/log4cplus-2.0.7/build/install/usr/local/lib/liblog4cplus.so.3" ./
    cp ${build_path}"/libMonkeyGL.so" ./
    cp ${py_build_path}"/pyMonkeyGL.so" ./

    cd ..
    makesure_folder ${install_path}/"include"
    cd ${install_path}/"include"
    rm -rf ./*
    cp -r ${build_path}"/ITK-5.2.1/build/install/include/ITK-5.2" ./
    cp -r ${build_path}"/log4cplus-2.0.7/build/install/usr/local/include/log4cplus" ./
    cp ${source_path}"/core/Defines.h" ./
    cp ${source_path}"/core/DeviceInfo.h" ./
    cp ${source_path}"/core/Direction.h" ./
    cp ${source_path}"/core/HelloMonkey.h" ./
    cp ${source_path}"/core/PlaneInfo.h" ./
    cp ${source_path}"/core/Point.h" ./
}

if [ $1 == "cpp" ]; then
    build_cpp
elif [ $1 == "itk" ]; then
    build_itk_lib
elif [ $1 == "log" ]; then
    build_log_lib
elif [ $1 == "pybind" ]; then
    build_pybind
elif [ $1 == "all" ]; then
    build_cpp
    build_pybind
elif [ $1 == "install" ]; then
    install
else
    echo "invalid project"
fi