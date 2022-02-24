#include "DeviceInfo.h"
#include "Defines.h"

using namespace MonkeyGL;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

int* get_data(){
    int* pdata = new int[10];
    for (int i=0; i<10; i++){
        pdata[i] = 100+i;
    }
    return pdata;
}

PYBIND11_MODULE(pyMonkeyGL, m) {
    pybind11::class_<DeviceInfo>(m, "DeviceInfo")
        .def(pybind11::init<>())
        .def("GetCount", &DeviceInfo::GetCount);

    
    pybind11::class_<RGBA>(m, "RGBA")
        .def(pybind11::init<>())
        .def(pybind11::init<float, float, float, float>())
        .def("Print", &RGBA::Print);

    m.def("get_data", &get_data, pybind11::return_value_policy::reference);

}