平台：Nvidia Jetson Nano
OS:Ubuntu 18.04 desktop
CPU:Cortex-A57 (Arm64架构) /4G Memory  
GPU: Nvidia Tegra X1 Maxwell架构 (128Core)
GCC: 7.5.0

git clone https://github.com/jiwenchen/MonkeyGL
阅读REAMD.md，然后分别执行
MonkeyGL/build.sh cpp Debug
MonkeyGL/build.sh pybind Debug

中间遇到的问题和解决
1）cuda的安装
Install_scripts/cuda_install_ubuntu.sh失效了。
因为这个里面只支持amd64，而不支持arm64
需要从cuda toolkit中下载安装，最终安装了Cuda 10.2

2）cmake需要更新
原来系统中是3.10，且通过apt无法升级到更高版本
编译器需要3.16以上，手动从网站上下载了cmake 3.24，并设置了路径

3）./build.sh cpp Debug的时候出错

在cmake后，修改MonkeyGL/build/CMakeFiles/MonkeyGL.dir/flags.make和Links.txt
以及CMakeLists.txt，在这些文件中去掉 -msse4.1 -mpclmul，然后执行在MonkeyGL/build下执行make

最好的办法是在MonkeyGL/下运行find .|xargs | grep -ri ‘msse’，找到后全部改掉。

其中为了解决问题将build.sh中的-DSSE=1改为-DSSE=0，关掉了SSE开关。

4）解决make中的错误
narrowing conversion of '-2' from int to char
在编译以下文件的时候出错：
MonekeyGL/core/Base64.hpp
MonekeyGL/core/MarchingCube.cpp
网络上有一个解决方案是把char的定义改成signed char，避免编译器的判断。

5）执行./build.sh pybind Debug的时候，出错
pyMonkeyGL.so本来放在 MonkeyGL/目录上，但是在./build.sh pybind Debug的时候，需要在MonkeyGL/build下。
