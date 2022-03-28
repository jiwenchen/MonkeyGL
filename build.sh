argNum=$#
source_path=$(pwd)

build_path=${source_path}"/build"

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

build_cpp() {
    makesure_folder ${build_path}
    cd ${build_path}
    if [ ${build_type} == "Clean" ]; then
      echo "clean cpp build"
      rm -rf ./*
    else
      cmake ../ -DCMAKE_BUILD_TYPE=${build_type}
      make
    fi
}

build_pybind() {
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