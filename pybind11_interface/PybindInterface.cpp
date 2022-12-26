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

template<typename T>
std::shared_ptr<T> _arrays_2d_to_ptr(py::array_t<T> npData, int& nWidth, int& nHeight) {
    py::buffer_info buf = npData.request();
    nWidth = buf.shape[0];
    nHeight = buf.shape[1];
    int cnt = buf.size;

    T* ptr = (T*)buf.ptr;
    std::shared_ptr<T> pData(new T[cnt * sizeof(T)]);

    memcpy(pData.get(), ptr, cnt*sizeof(T));
    return pData;
}

std::vector<Point3d> _arrays_3d_to_points(py::array_t<float> cprLineArray) {
    std::vector<Point3d> cprLine;
    py::buffer_info buf = cprLineArray.request();
    int nLen = buf.shape[0];
    int nDim = buf.shape[1];
    if (nDim != 3){
        return cprLine;
    }
    int cnt = buf.size;
    float* ptr = (float*)buf.ptr;

    for (int i=0; i<nLen; i++)
    {
        cprLine.push_back(Point3d(ptr[3*i], ptr[3*i+1], ptr[3*i+2]));
    }
    return cprLine;
}

class pyHelloMonkey : public HelloMonkey {

public:
    virtual py::array_t<short> GetVolumeArray(){
        int nWidth=0, nHeight=0, nDepth=0;
        std::shared_ptr<short> pData = GetVolumeData(nWidth, nHeight, nDepth);
        return _ptr_to_arrays_3d(pData.get(), nWidth, nHeight, nDepth);
    };

    virtual bool SetVolumeArray(py::array_t<short> npData){
        int nWidth = 0;
        int nHeight = 0;
        int nDepth = 0;
        std::shared_ptr<short> pData = _arrays_3d_to_ptr(npData, nWidth, nHeight, nDepth);
        return SetVolumeData(pData, nWidth, nHeight, nDepth);
    };

    virtual void Transfer2Base64Array(py::array_t<unsigned char> npData){
        int nWidth = 0;
        int nHeight = 0;
        std::shared_ptr<unsigned char> pData = _arrays_2d_to_ptr(npData, nWidth, nHeight);
        return Transfer2Base64(pData.get(), nWidth, nHeight);
    };

    virtual unsigned char AddNewObjectMaskArray(py::array_t<unsigned char> npData){
        int nWidth = 0;
        int nHeight = 0;
        int nDepth = 0;
        std::shared_ptr<unsigned char> pData = _arrays_3d_to_ptr(npData, nWidth, nHeight, nDepth);
        return AddNewObjectMask(pData, nWidth, nHeight, nDepth);
    };

    virtual bool UpdateMaskArray(py::array_t<unsigned char> npData, const unsigned char& nLabel){
        int nWidth = 0;
        int nHeight = 0;
        int nDepth = 0;
        std::shared_ptr<unsigned char> pData = _arrays_3d_to_ptr(npData, nWidth, nHeight, nDepth);
        return UpdateObjectMask(pData, nWidth, nHeight, nDepth, nLabel);
    };

    virtual py::array_t<unsigned char> GetVRArray(int nWidth, int nHeight){
	    std::shared_ptr<unsigned char> pVR;
        GetVRData(pVR, nWidth, nHeight);
        return _ptr_to_arrays_3d((unsigned char*)pVR.get(), 3, nWidth, nHeight);
    }
    
    virtual py::array_t<uint8_t> GetVRDataArray_png(int nWidth, int nHeight){
        std::vector<uint8_t> out_buf = GetVRData_png(nWidth, nHeight);
        return _ptr_to_arrays_1d(out_buf.data(), out_buf.size());
    }

    virtual bool SetCPRLinePatientArray(py::array_t<float> cprLineArray){
        std::vector<Point3d> cprLine = _arrays_3d_to_points(cprLineArray);
        return SetCPRLinePatient(cprLine);
    }

    virtual bool SetCPRLineVoxelArray(py::array_t<float> cprLineArray){
        std::vector<Point3d> cprLine = _arrays_3d_to_points(cprLineArray);
        return SetCPRLineVoxel(cprLine);
    }

    virtual int GetPlaneCurrentIndex(PlaneType planeType){
        int index = -1;
        GetPlaneIndex(index, planeType);
        return index;
    }

    virtual int GetPlaneTotalNumber(PlaneType planeType){
        int nTotalNum = 0;
        GetPlaneNumber(nTotalNum, planeType);
        return nTotalNum;
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
        .value("PlaneStretchedCPR", PlaneType::PlaneStretchedCPR)
        .value("PlaneStraightenedCPR", PlaneType::PlaneStraightenedCPR)
        .export_values();

    py::enum_<Orientation>(m, "Orientation")
        .value("OrientationNotDefined", Orientation::OrientationNotDefined)
        .value("OrientationAnterior", Orientation::OrientationAnterior)
        .value("OrientationPosterior", Orientation::OrientationPosterior)
        .value("OrientationLeft", Orientation::OrientationLeft)
        .value("OrientationRight", Orientation::OrientationRight)
        .value("OrientationHead", Orientation::OrientationHead)
        .value("OrientationFoot", Orientation::OrientationFoot)
        .export_values();

    py::enum_<MPRType>(m, "MPRType")
        .value("MPRTypeNotDefined", MPRType::MPRTypeNotDefined)
        .value("MPRTypeAverage", MPRType::MPRTypeAverage)
        .value("MPRTypeMIP", MPRType::MPRTypeMIP)
        .value("MPRTypeMinIP", MPRType::MPRTypeMinIP)
        .export_values();

    py::enum_<RenderType>(m, "RenderType")
        .value("RenderTypeNotDefined", RenderType::RenderTypeNotDefined)
        .value("RenderTypeVR", RenderType::RenderTypeVR)
        .value("RenderTypeMIP", RenderType::RenderTypeMIP)
        .value("RenderTypeSurface", RenderType::RenderTypeSurface)
        .export_values();

    py::enum_<LayerType>(m, "LayerType")
        .value("LayerTypeNotDefined", LayerType::LayerTypeNotDefined)
        .value("LayerTypeImage", LayerType::LayerTypeImage)
        .value("LayerTypeAnnotation", LayerType::LayerTypeAnnotation)
        .value("LayerTypeCPRLine", LayerType::LayerTypeCPRLine)
        .export_values();

    py::enum_<FontSize>(m, "FontSize")
        .value("FontSizeNotDefined", FontSize::FontSizeNotDefined)
        .value("FontSizeSmall", FontSize::FontSizeSmall)
        .value("FontSizeMiddle", FontSize::FontSizeMiddle)
        .value("FontSizeBig", FontSize::FontSizeBig)
        .export_values();

    py::enum_<AnnotationFormat>(m, "AnnotationFormat")
        .value("AnnotationFormatNotDefined", AnnotationFormat::AnnotationFormatNotDefined)
        .value("AnnotationFormatLeft", AnnotationFormat::AnnotationFormatLeft)
        .value("AnnotationFormatRight", AnnotationFormat::AnnotationFormatRight)
        .value("AnnotationFormatCenter", AnnotationFormat::AnnotationFormatCenter)
        .export_values();

    py::class_<DeviceInfo>(m, "DeviceInfo")
        .def(py::init<>())
        .def("GetCount", &DeviceInfo::GetCount);

    py::class_<RGB>(m, "RGB")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def("Print", &RGB::Print);
    
    py::class_<RGBA>(m, "RGBA")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def("Print", &RGBA::Print);

    py::class_<Direction3d>(m, "Direction3d")
        .def(py::init<>())
        .def(py::init<double, double, double>());

    py::class_<Point2d>(m, "Point2d")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def("x", &Point2d::x)
        .def("y", &Point2d::y)
        .def("SetX", &Point2d::SetX)
        .def("SetY", &Point2d::SetY);

    py::class_<Point3d>(m, "Point3d")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def("x", &Point3d::x)
        .def("y", &Point3d::y)
        .def("z", &Point3d::z)
        .def("SetX", &Point3d::SetX)
        .def("SetY", &Point3d::SetY)
        .def("SetZ", &Point3d::SetZ);

    py::class_<pyHelloMonkey>(m, "HelloMonkey")
        .def(py::init<>())
        .def("SetLogLevel", &pyHelloMonkey::SetLogLevel)
        .def("LoadVolumeFile", &pyHelloMonkey::LoadVolumeFile)
        .def("SetVolumeArray", &pyHelloMonkey::SetVolumeArray)
        .def("AddObjectMaskFile", &pyHelloMonkey::AddObjectMaskFile)
        .def("AddNewObjectMaskArray", &pyHelloMonkey::AddNewObjectMaskArray)
        .def("UpdateMaskArray", &pyHelloMonkey::UpdateMaskArray)
        .def("SetSpacing", &pyHelloMonkey::SetSpacing)
        .def("SetDirection", &pyHelloMonkey::SetDirection)
        .def("SetOrigin", &pyHelloMonkey::SetOrigin)
        .def("SetTransferFunc", static_cast<bool (pyHelloMonkey::*)(std::map<int, RGBA>)>(&pyHelloMonkey::SetTransferFunc))
        .def("SetTransferFunc", static_cast<bool (pyHelloMonkey::*)(std::map<int, RGBA>, unsigned char)>(&pyHelloMonkey::SetTransferFunc))
        .def("SetTransferFunc", static_cast<bool (pyHelloMonkey::*)(std::map<int, RGBA>, std::map<int, float>)>(&pyHelloMonkey::SetTransferFunc))
        .def("SetTransferFunc", static_cast<bool (pyHelloMonkey::*)(std::map<int, RGBA>, std::map<int, float>, unsigned char)>(&pyHelloMonkey::SetTransferFunc))
        .def("LoadTransferFunction", &pyHelloMonkey::LoadTransferFunction)
        .def("SaveTransferFunction", &pyHelloMonkey::SaveTransferFunction)
        .def("SetColorBackground", &pyHelloMonkey::SetColorBackground)
        .def("Reset", &pyHelloMonkey::Reset)
        .def("Anterior", &pyHelloMonkey::Anterior)
        .def("Posterior", &pyHelloMonkey::Posterior)
        .def("Left", &pyHelloMonkey::Left)
        .def("Right", &pyHelloMonkey::Right)
        .def("Head", &pyHelloMonkey::Head)
        .def("Foot", &pyHelloMonkey::Foot)
        .def("SetVRWWWL", static_cast<bool (pyHelloMonkey::*)(float, float)>(&pyHelloMonkey::SetVRWWWL))
        .def("SetVRWWWL", static_cast<bool (pyHelloMonkey::*)(float, float, unsigned char)>(&pyHelloMonkey::SetVRWWWL))
        .def("SetObjectAlpha", static_cast<bool (pyHelloMonkey::*)(float)>(&pyHelloMonkey::SetObjectAlpha))
        .def("SetObjectAlpha", static_cast<bool (pyHelloMonkey::*)(float, unsigned char)>(&pyHelloMonkey::SetObjectAlpha))
        .def("SetRenderType", &pyHelloMonkey::SetRenderType)
        .def("Rotate", &pyHelloMonkey::Rotate)
        .def("Pan", &pyHelloMonkey::Pan)
        .def("Zoom", &pyHelloMonkey::Zoom)
        .def("UpdateThickness", &pyHelloMonkey::UpdateThickness)
        .def("SetThickness", &pyHelloMonkey::SetThickness)
        .def("GetThickness", &pyHelloMonkey::GetThickness)
        .def("SetMPRType", &pyHelloMonkey::SetMPRType)
        .def("Browse", &pyHelloMonkey::Browse)
        .def("PanCrossHair", &pyHelloMonkey::PanCrossHair)
        .def("RotateCrossHair", &pyHelloMonkey::RotateCrossHair)
        .def("SetCPRLinePatientArray", &pyHelloMonkey::SetCPRLinePatientArray)
        .def("SetCPRLineVoxelArray", &pyHelloMonkey::SetCPRLineVoxelArray)
        .def("RotateCPR", &pyHelloMonkey::RotateCPR)
        .def("SetCPRLineColor", &pyHelloMonkey::SetCPRLineColor)
        .def("SetVRSize", &pyHelloMonkey::SetVRSize)
        .def("ShowPlaneInVR", &pyHelloMonkey::ShowPlaneInVR)
        .def("SetPlaneIndex", &pyHelloMonkey::SetPlaneIndex)
        .def("Transfer2Base64Array", &pyHelloMonkey::Transfer2Base64Array)
        .def("AddAnnotation", &pyHelloMonkey::AddAnnotation)
        .def("RemovePlaneAnnotations", &pyHelloMonkey::RemovePlaneAnnotations)
        .def("RemoveAllAnnotations", &pyHelloMonkey::RemoveAllAnnotations)
        .def("EnableLayer", &pyHelloMonkey::EnableLayer)

        .def("GetZoomRatio", &pyHelloMonkey::GetZoomRatio)
        .def("GetPlaneCurrentIndex", &pyHelloMonkey::GetPlaneCurrentIndex)
        .def("GetPlaneTotalNumber", &pyHelloMonkey::GetPlaneTotalNumber)
        .def("GetVolumeArray", &pyHelloMonkey::GetVolumeArray)
        .def("GetVRArray", &pyHelloMonkey::GetVRArray)
        .def("GetVRData_pngString", &pyHelloMonkey::GetVRData_pngString)
        .def("GetVRData_png", &pyHelloMonkey::GetVRData_png)
        .def("SaveVR2Png", &pyHelloMonkey::SaveVR2Png)
        .def("GetPlaneData_pngString", &pyHelloMonkey::GetPlaneData_pngString)
        .def("GetOriginData_pngString", &pyHelloMonkey::GetOriginData_pngString)
        .def("GetCrossHairPoint", static_cast<Point2d (pyHelloMonkey::*)(const PlaneType&)>(&pyHelloMonkey::GetCrossHairPoint))
        .def("GetPlaneIndex", static_cast<int (pyHelloMonkey::*)(const PlaneType&)>(&pyHelloMonkey::GetPlaneIndex))
        .def("GetPlaneNumber", static_cast<int (pyHelloMonkey::*)(const PlaneType&)>(&pyHelloMonkey::GetPlaneNumber));
}