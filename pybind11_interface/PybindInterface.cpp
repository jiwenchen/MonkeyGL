#include "DeviceInfo.h"
#include "Defines.h"
#include "HelloMonkey.h"

using namespace MonkeyGL;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

int* get_data(){
    int* pdata = new int[10];
    for (int i=0; i<10; i++){
        pdata[i] = 100+i;
    }
    return pdata;
}

bool change(std::map<std::string, int>& val){
    val["width"] = 10;
    val["height"] = 20;
    return true;
}

namespace py = pybind11;
py::array_t<double> add_arrays_3d(py::array_t<double>& input1, py::array_t<double>& input2) {
    auto r1 = input1.unchecked<3>();
    auto r2 = input2.unchecked<3>();

    py::array_t<double> out = py::array_t<double>(input1.size());
    out.resize({ input1.shape()[0], input1.shape()[1], input1.shape()[2] });
    auto r3 = out.mutable_unchecked<3>();

    for (int i = 0; i < input1.shape()[0]; i++)
    {
        for (int j = 0; j < input1.shape()[1]; j++)
        {
            for (int k = 0; k < input1.shape()[2]; k++)
            {
                double value1 = r1(i, j, k);
                double value2 = r2(i, j, k);
                r3(i, j, k) = value1 + value2;
            
            }
        }
    }
    return out;
}


template<typename T>
py::array_t<T> _ptr_to_arrays_1d(T* data, py::ssize_t col) {
    auto result = py::array_t<T>(col);
    py::buffer_info buf = result.request();
    T* ptr = (T*)buf.ptr;
 
    for (auto i = 0; i < col; i++) 
        ptr[i] = data[i];
    
    return result;
}


template<typename T>
py::array_t<T> _ptr_to_arrays_3d(T* data, py::ssize_t chunk, py::ssize_t row, py::ssize_t col) {
    auto result = _ptr_to_arrays_1d(data, chunk * row * col);
    result.resize({ col, row, chunk });
    return result;
}

class HelloMonkeyInterface : public HelloMonkey {

public:
    py::array_t<short> GetVolumeArray(){
        short* pData = GetVolumeData();
        return _ptr_to_arrays_1d(pData, 100);
    };

    virtual py::array_t<unsigned char> GetVRArray(int nWidth, int nHeight){
	    std::shared_ptr<unsigned char> pVR (new unsigned char[nWidth*nHeight*3]);
        GetVRData((unsigned char*)pVR.get(), nWidth, nHeight);
        return _ptr_to_arrays_3d((unsigned char*)pVR.get(), 3, nHeight, nWidth);
    }
};

PYBIND11_MODULE(pyMonkeyGL, m) {

    py::class_<HelloMonkeyInterface>(m, "HelloMonkey")
        .def(py::init<>())
        .def("SetVolumeFile", &HelloMonkeyInterface::SetVolumeFile)
        .def("SetAnisotropy", &HelloMonkeyInterface::SetAnisotropy)
        .def("SetDirection", &HelloMonkeyInterface::SetDirection)
        .def("SetTransferFunc", static_cast<void (HelloMonkeyInterface::*)(const std::map<int, RGBA>&)>(&HelloMonkeyInterface::SetTransferFunc))
        .def("SetTransferFunc", static_cast<void (HelloMonkeyInterface::*)(const std::map<int, RGBA>&, const std::map<int, double>&)>(&HelloMonkeyInterface::SetTransferFunc))
        .def("Reset", &HelloMonkeyInterface::Reset)
        .def("Rotate", &HelloMonkeyInterface::Rotate)
        .def("GetVolumeArray", &HelloMonkeyInterface::GetVolumeArray)
        .def("SaveVR2BMP", &HelloMonkeyInterface::SaveVR2BMP)
        .def("GetVRArray", &HelloMonkeyInterface::GetVRArray);
        

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def(py::init<>())
        .def("GetCount", &DeviceInfo::GetCount);

    
    py::class_<RGBA>(m, "RGBA")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def("Print", &RGBA::Print);

    m.def("get_data", &get_data, py::return_value_policy::reference);
    m.def("change", &change);

    m.def("add_arrays_3d", &add_arrays_3d);
}