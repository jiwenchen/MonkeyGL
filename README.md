# MonkeyGL

## install cuda
### linux (tested in Ubuntu)
>./install/cuda_install_ubuntu.sh

## build project
### linux (tested in Ubuntu)
>mkdir build  
>cd build  
>cmake ../  
>make --> c++ shared library: libMonkeyGL.so  

>cd ./pybind11_interface  
>mkdir build  
>cd build  
>cmake ../  
>make --> pybind11 shared library: pyMonkeyGL.so