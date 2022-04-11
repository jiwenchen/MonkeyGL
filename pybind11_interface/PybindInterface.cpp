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
py::array_t<T> _ptr_to_arrays_1d(T* data, py::ssize_t cnt) {
    auto result = py::array_t<T>(cnt);
    py::buffer_info buf = result.request();
    T* ptr = (T*)buf.ptr;

    memcpy(ptr, data, cnt*sizeof(T));
    return result;
}

template<typename T>
std::shared_ptr<T> _arrays_1d_to_ptr(py::array_t<T> npData, int& cnt) {
    py::buffer_info buf = npData.request();
    cnt = buf.size;

    T* ptr = (T*)buf.ptr;
    std::shared_ptr<T> pData(new T[cnt * sizeof(T)]);

    memcpy(pData.get(), ptr, cnt*sizeof(T));
    return pData;
}

template<typename T>
py::array_t<T> _ptr_to_arrays_3d(T* data, py::ssize_t width, py::ssize_t height, py::ssize_t depth) {
    auto result = _ptr_to_arrays_1d(data, width * height * depth);
    result.resize({ width, height, depth });
    return result;
}

template<typename T>
std::shared_ptr<T> _arrays_3d_to_ptr(py::array_t<T> npData, int& nWidth, int& nHeight, int& nDepth) {
    py::buffer_info buf = npData.request();
    nWidth = buf.shape[0];
    nHeight = buf.shape[1];
    nDepth = buf.shape[2];
    int cnt = buf.size;

    T* ptr = (T*)buf.ptr;
    std::shared_ptr<T> pData(new T[cnt * sizeof(T)]);

    memcpy(pData.get(), ptr, cnt*sizeof(T));
    return pData;
}

class pyHelloMonkey : public HelloMonkey {

public:
    py::array_t<short> GetVolumeArray(){
        int nWidth=0, nHeight=0, nDepth=0;
        std::shared_ptr<short> pData = GetVolumeData(nWidth, nHeight, nDepth);
        return _ptr_to_arrays_3d(pData.get(), nWidth, nHeight, nDepth);
    };

    bool SetVolumeArray(py::array_t<short> npData){
        int nWidth = 0;
        int nHeight = 0;
        int nDepth = 0;
        std::shared_ptr<short> pData = _arrays_3d_to_ptr(npData, nWidth, nHeight, nDepth);
        return SetVolumeData(pData, nWidth, nHeight, nDepth);
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

    virtual void SetColorBackgroundArray(py::array_t<float> clrBkg){
        int cnt = 0;
        std::shared_ptr<float> pData = _arrays_1d_to_ptr(clrBkg, cnt);
        SetColorBackground(pData.get(), cnt);
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

    py::class_<pyHelloMonkey>(m, "HelloMonkey")
        .def(py::init<>())
        .def("SetLogLevel", &pyHelloMonkey::SetLogLevel)
        .def("SetVolumeFile", &pyHelloMonkey::SetVolumeFile)
        .def("SetVolumeArray", &pyHelloMonkey::SetVolumeArray)
        .def("SetAnisotropy", &pyHelloMonkey::SetAnisotropy)
        .def("SetDirection", &pyHelloMonkey::SetDirection)
        .def("SetTransferFunc", static_cast<void (pyHelloMonkey::*)(const std::map<int, RGBA>&)>(&pyHelloMonkey::SetTransferFunc))
        .def("SetTransferFunc", static_cast<void (pyHelloMonkey::*)(const std::map<int, RGBA>&, const std::map<int, double>&)>(&pyHelloMonkey::SetTransferFunc))
        .def("SetColorBackgroundArray", &pyHelloMonkey::SetColorBackgroundArray)
        .def("Reset", &pyHelloMonkey::Reset)
        .def("SetVRWWWL", &pyHelloMonkey::SetVRWWWL)
        .def("Rotate", &pyHelloMonkey::Rotate)
        .def("Browse", &pyHelloMonkey::Browse)
        .def("UpdateThickness", &pyHelloMonkey::UpdateThickness)
        .def("SetMPRType", &pyHelloMonkey::SetMPRType)

        .def("GetVolumeArray", &pyHelloMonkey::GetVolumeArray)
        .def("GetVRArray", &pyHelloMonkey::GetVRArray)
        .def("GetVRData_pngString", &pyHelloMonkey::GetVRData_pngString)
        .def("GetVRData_png", &pyHelloMonkey::GetVRData_png)
        .def("SaveVR2Png", &pyHelloMonkey::SaveVR2Png)
        .def("GetPlaneData_pngString", &pyHelloMonkey::GetPlaneData_pngString);
}