// MIT License

// Copyright (c) 2022 jiwenchen(cjwbeyond@hotmail.com)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "DeviceInfo.h"
#include "Defines.h"
#include "HelloMonkey.h"
#include "Direction.h"

using namespace MonkeyGL;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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

    virtual py::array_t<uint8_t> GetVRDataArray_png(int nWidth, int nHeight){
        std::vector<uint8_t> out_buf = GetVRData_png(nWidth, nHeight);
        return _ptr_to_arrays_1d(out_buf.data(), out_buf.size());
    }
};

PYBIND11_MODULE(pyMonkeyGL, m) {

    py::enum_<LogLevel>(m, "LogLevel")
        .value("LogLevelNotDefined", LogLevel::LogLevelNotDefined)
        .value("LogLevelInfo", LogLevel::LogLevelInfo)
        .value("LogLevelWarn", LogLevel::LogLevelWarn)
        .value("LogLevelError", LogLevel::LogLevelError)
        .export_values();

    py::enum_<PlaneType>(m, "PlaneType")
        .value("PlaneNotDefined", PlaneType::PlaneNotDefined)
        .value("PlaneAxial", PlaneType::PlaneAxial)
        .value("PlaneSagittal", PlaneType::PlaneSagittal)
        .value("PlaneCoronal", PlaneType::PlaneCoronal)
        .value("PlaneAxialOblique", PlaneType::PlaneAxialOblique)
        .value("PlaneSagittalOblique", PlaneType::PlaneSagittalOblique)
        .value("PlaneCoronalOblique", PlaneType::PlaneCoronalOblique)
        .value("PlaneVR", PlaneType::PlaneVR)
        .export_values();

    py::enum_<MPRType>(m, "MPRType")
        .value("MPRTypeNotDefined", MPRType::MPRTypeNotDefined)
        .value("MPRTypeAverage", MPRType::MPRTypeAverage)
        .value("MPRTypeMIP", MPRType::MPRTypeMIP)
        .value("MPRTypeMinIP", MPRType::MPRTypeMinIP)
        .export_values();

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def(py::init<>())
        .def("GetCount", &DeviceInfo::GetCount);
    
    py::class_<RGBA>(m, "RGBA")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def("Print", &RGBA::Print);

    py::class_<Direction3d>(m, "Direction3d")
        .def(py::init<>())
        .def(py::init<float, float, float>());

    py::class_<HelloMonkeyInterface>(m, "HelloMonkey")
        .def(py::init<>())
        .def("SetLogLevel", &HelloMonkeyInterface::SetLogLevel)
        .def("SetVolumeFile", &HelloMonkeyInterface::SetVolumeFile)
        .def("SetAnisotropy", &HelloMonkeyInterface::SetAnisotropy)
        .def("SetDirection", &HelloMonkeyInterface::SetDirection)
        .def("SetTransferFunc", static_cast<void (HelloMonkeyInterface::*)(const std::map<int, RGBA>&)>(&HelloMonkeyInterface::SetTransferFunc))
        .def("SetTransferFunc", static_cast<void (HelloMonkeyInterface::*)(const std::map<int, RGBA>&, const std::map<int, double>&)>(&HelloMonkeyInterface::SetTransferFunc))
        .def("Reset", &HelloMonkeyInterface::Reset)
        .def("SetVRWWWL", &HelloMonkeyInterface::SetVRWWWL)
        .def("Rotate", &HelloMonkeyInterface::Rotate)
        .def("Browse", &HelloMonkeyInterface::Browse)
        .def("UpdateThickness", &HelloMonkeyInterface::UpdateThickness)
        .def("SetMPRType", &HelloMonkeyInterface::SetMPRType)

        .def("GetVolumeArray", &HelloMonkeyInterface::GetVolumeArray)
        .def("GetVRArray", &HelloMonkeyInterface::GetVRArray)
        .def("GetVRData_pngString", &HelloMonkeyInterface::GetVRData_pngString)
        .def("GetVRData_png", &HelloMonkeyInterface::GetVRData_png)
        .def("GetPlaneData_pngString", &HelloMonkeyInterface::GetPlaneData_pngString);
}