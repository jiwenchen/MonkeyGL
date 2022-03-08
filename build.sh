argNum=$#
source_path=$(pwd)

build_path=${source_path}"/build"

py_build_path=${source_path}"/pybind11_interface/build"

build_type="Release"

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

if [ $2 == "Debug" ]; then
    build_type="Debug"
fi

build_cpp() {
    makesure_folder ${build_path}
    cd ${build_path}
    cmake ../ -DCMAKE_BUILD_TYPE=${build_type}
    make
}

build_pybind() {
    makesure_folder ${py_build_path}
    cd ${py_build_path}
    cmake ../ -DCMAKE_BUILD_TYPE=${build_type}
    make
}

if [ $1 == "cpp" ]; then
    build_cpp
elif [ $1 == "pybind" ]; then
    build_pybind
elif [ $1 == "all" ]; then
    build_cpp
    build_pybind
else
    echo "invalid project"
fi