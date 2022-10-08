**MonkeyGL** is a great open source fast image rendering library base on CUDA and ITK.   
It reduces dependency on other GL library like OpenGL, and provides a python interface to improve usability.   
Following text focuses on building MonkeyGL on arm64 CPU platform to prove its platform compatibility.  
  
# Hardware and software environment
Hardware：Nvidia Jetson Nano  
OS: Ubuntu 18.04 desktop  
CPU: Cortex-A57 (Arm64) /4G Memory   
GPU: Nvidia Tegra X1 Maxwell架构 (128Core)  
GCC: 7.5.0 
  
# execute build command
git clone https://github.com/jiwenchen/MonkeyGL  
following REAMD.md，then execute two command in sequence:    
1. `MonkeyGL/build.sh cpp Debug`  
2. `MonkeyGL/build.sh pybind Debug` 
  
# Problem solving during build
## cuda installation
Install_scripts/cuda_install_ubuntu.sh does not work on Arm64 platform.  
By default, `sudo apt-get install cuda` seems to support amd64 only.  
Go to CUDA toolkit website and download binary to finish CUDA installation.  
[CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) is installed.  

## upgrade cmake
In ubuntu18.04, cmake 3.10 is shipped. MonkeyGL need at least 3.16 or higher.  
Go to cmake official site and download binary.  
set a symbolic link from `/usr/bin/cmake` to newly downloaded cmake.    
[cmake 3.24](https://cmake.org/download/) is used.   

## No cpp switch in `./build.sh cpp Debug`
During executing "./build.sh cpp Debug", encounter following errors：  
1. `C++ error: unrecognized command line option '-msse4.1'` 
2. `C++ error: unrecognized command line option '-mpclmul'`  
 
It's very possible that these two swiches are not supported in arm64 architecture.   
Use `find .|xargs | grep -ri ‘msse’` to search all possible occurrence and remove them. 
These switches are found in flags.make, links.txt and CMakeLists.txt.
In build.sh, change `-DSSE=1` to be `-DSSE=0`, and close SSE switch.
  
## problem in `make`
During 'make', a problem is found with `narrowing conversion of '-2' from int to char`   
There is a solution, that is change `char []` to be `singed char []`.  
After fixing following two files, 'make' can be done.  
1. MonekeyGL/core/Base64.hpp  
2. MonekeyGL/core/MarchingCube.cpp   

But **NOT very sure** of its impact to result of calculation.
  
## problem in `./build.sh pybind Debug`
pyMonkeyGL.so should be in `MonkeyGL/build` directory. But actually it's in `MonkeyGL/`。  
You need to manually copy file to target directory.
  
Finally build succeed! Let's enjoy MonkeyGL together.
