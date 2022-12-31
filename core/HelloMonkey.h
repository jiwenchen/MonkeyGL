// MIT License

// Copyright (c) 2022-2023 jiwenchen(cjwbeyond@hotmail.com)

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

#pragma once
#include <map>
#include <vector>
#include <memory>
#include "Defines.h"
#include "Direction.h"
#include "PlaneInfo.h"
#include "BatchInfo.h"

namespace MonkeyGL {

    class IRender;

    class HelloMonkey
    {
    public:
        HelloMonkey();
        ~HelloMonkey(void);

    public:
        virtual void SetLogLevel(LogLevel level);
        virtual void LoadVolumeFile(const char* szFile);
        virtual void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        virtual void SetSpacing(double x, double y, double z);
        virtual void SetOrigin(Point3d pt);
        virtual void Reset();
        virtual void SetColorBackground(RGBA clrBG);
        virtual bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);
        virtual unsigned char AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth);
        virtual unsigned char AddObjectMaskFile(const char* szFile);
        virtual bool UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel);

    // output
        virtual std::shared_ptr<short> GetVolumeData(int& nWidth, int& nHeight, int& nDepth);
        virtual bool GetPlaneMaxSize(int& nWidth, int& nHeight, const PlaneType& planeType);
        virtual bool GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);
        virtual std::string GetPlaneData_pngString(const PlaneType& planeType);
        virtual std::string GetOriginData_pngString(int slice);

        virtual bool GetCrossHairPoint(double& x, double& y, const PlaneType& planeType);
        virtual Point2d GetCrossHairPoint(const PlaneType& planeType);
        virtual bool TransferImage2Voxel(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType);
        virtual bool GetCrossHairPoint3D(Point3d& pt);
        virtual bool GetDirection(Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType);
        virtual bool GetDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType);
        virtual bool GetBatchDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType);

        virtual double GetPixelSpacing(PlaneType planeType);

        virtual bool SetVRSize(int nWidth, int nHeight);
        virtual bool GetVRData(std::shared_ptr<unsigned char>& pData, int nWidth, int nHeight);
        virtual std::string GetVRData_pngString();
        virtual std::vector<uint8_t> GetVRData_png(int nWidth, int nHeight);
        virtual void SaveVR2Png(const char* szFile, int nWidth, int nHeight);

        virtual bool GetBatchData(std::vector<short*>& vecBatchData, const BatchInfo& batchInfo);

        virtual bool GetPlaneIndex(int& index, const PlaneType&  planeType);
        virtual int GetPlaneIndex(const PlaneType&  planeType);
        virtual bool GetPlaneNumber(int& nTotalNum, const PlaneType&  planeType);
        virtual int GetPlaneNumber(const PlaneType&  planeType);
   
        virtual bool GetPlaneRotateMatrix(float* pMatrix, PlaneType planeType);

    // interactions
        virtual void Anterior();
        virtual void Posterior();
        virtual void Left();
        virtual void Right();
        virtual void Head();
        virtual void Foot();

        virtual void SetRenderType(RenderType type);
        virtual void Rotate(float fxRotate, float fyRotate);
        virtual float Zoom(float ratio);
        virtual float GetZoomRatio();
        virtual void Pan(float fxShift, float fyShift);
        virtual bool SetVRWWWL(float fWW, float fWL);
        virtual bool SetVRWWWL(float fWW, float fWL, unsigned char nLabel);
        virtual bool SetObjectAlpha(float fAlpha);
        virtual bool SetObjectAlpha(float fAlpha, unsigned char nLabel);
        virtual bool SetTransferFunc(std::map<int, RGBA> ctrlPoints);
        virtual bool SetTransferFunc(std::map<int, RGBA> ctrlPoints, unsigned char nLabel);
        virtual bool SetTransferFunc(std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints);
        virtual bool SetTransferFunc(std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints, unsigned char nLabel);
        virtual bool LoadTransferFunction(const char* szFile);
        virtual bool SaveTransferFunction(const char* szFile);

        virtual void Browse(float fDelta, PlaneType planeType);	
        virtual void PanCrossHair(float fx, float fy, PlaneType planeType);
        virtual void RotateCrossHair(float fAngle, PlaneType planeType);
        virtual void SetPlaneIndex(int index, PlaneType planeType);

        virtual void UpdateThickness(double val);
        virtual void SetThickness(double val, PlaneType planeType);
        virtual double GetThickness(PlaneType planeType);
        virtual void SetMPRType(MPRType type);

        virtual bool AddAnnotation(PlaneType planeType, std::string txt, int x, int y, FontSize fontSize, AnnotationFormat annoFormat, RGBA clr);
        virtual bool RemovePlaneAnnotations(PlaneType planeType);
        virtual bool RemoveAllAnnotations();
        virtual bool EnableLayer(LayerType layerType, bool bEnable);

        // cpr
        virtual bool SetCPRLinePatient(std::vector<Point3d> cprLine);
        virtual bool SetCPRLineVoxel(std::vector<Point3d> cprLine);
        virtual bool RotateCPR(float angle, PlaneType planeType);
        virtual void SetCPRLineColor(RGBA clr);

        // plane in VR
        virtual void ShowPlaneInVR(bool bShow);

        virtual void Transfer2Base64(unsigned char* pData, int nWidth, int nHeight);

    private:
        std::shared_ptr<IRender> m_pRender;
    };
}
