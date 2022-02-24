#include <iostream>
#include "DeviceInfo.h"
#include "Defines.h"

using namespace MonkeyGL;

int add(int a, int b){
    int c = a+b;
    std::cout << c << std::endl;
    return c;
}

#include <pybind11/pybind11.h>

PYBIND11_MODULE(example, m) {

    pybind11::class_<DeviceInfo>(m, "DeviceInfo")
        .def(pybind11::init<>())
        .def("GetCount", &DeviceInfo::GetCount);
        
    m.def("add", &add);
}