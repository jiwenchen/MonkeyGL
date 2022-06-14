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

#pragma once
#include <map>
#include "Defines.h"
#include "VolumeInfo.h"
#include "TransferFunction.h"
#include "Point.h"
#include "Direction.h"
#include "DataManager.h"
#include "BatchInfo.h"

namespace MonkeyGL{
    
    class IRender
    {
    public:
        IRender(void);
        virtual ~IRender(void);

    public:
    // volume info
        virtual bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);
        virtual unsigned char AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth);
        bool UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel);
        virtual void SetVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        virtual void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        virtual void SetSpacing(double x, double y, double z);
        virtual void SetOrigin(Point3d pt);
        virtual void Reset();
        virtual void SetColorBackground(RGBA clrBG);

        virtual void Browse(float fDelta, PlaneType planeType);
        virtual void SetPlaneIndex(int index, PlaneType planeType);
        virtual void PanCrossHair(int nx, int ny, PlaneType planeType);
        virtual void RotateCrossHair(float fAngle, PlaneType planeType);
        virtual void UpdateThickness(double val);
        virtual void SetThickness(double val, PlaneType planeType);
        virtual bool GetThickness(double& val, PlaneType planeType);
        virtual void SetMPRType(MPRType type);

        virtual bool TransferVoxel2ImageInVR(float& fx, float& fy, int nWidth, int nHeight, Point3d ptVoxel) = 0;

    // output
        virtual std::shared_ptr<short> GetVolumeData(int& nWidth, int& nHeight, int& nDepth);
        virtual std::shared_ptr<unsigned char> GetMaskData();
        virtual bool GetPlaneMaxSize(int& nWidth, int& nHeight, const PlaneType& planeType);
        virtual bool GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);

        virtual bool GetCrossHairPoint(double& x, double& y, const PlaneType& planeType) = 0;
        virtual bool TransferImage2Voxel(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType);
        virtual bool GetCrossHairPoint3D(Point3d& pt);
        virtual bool GetDirection(Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType);
        virtual bool GetDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType);
        virtual bool GetBatchDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType);

        virtual double GetPixelSpacing(PlaneType planeType);

        virtual bool GetVRData(unsigned char* pVR, int nWidth, int nHeight) = 0;

        virtual bool GetBatchData( std::vector<short*>& vecBatchData, BatchInfo batchInfo ) = 0;

        virtual bool GetPlaneIndex(int& index, PlaneType planeType);
        virtual bool GetPlaneNumber(int& nTotalNum, PlaneType planeType);
        virtual bool GetPlaneRotateMatrix(float* pMatirx, PlaneType planeType);


        virtual void Anterior() = 0;
        virtual void Posterior() = 0;
        virtual void Left() = 0;
        virtual void Right() = 0;
        virtual void Head() = 0;
        virtual void Foot() = 0;

        virtual void Rotate(float fxRotate, float fyRotate) = 0;
        virtual void Zoom(float ratio) = 0;
        virtual void Pan(float fxShift, float fyShift) = 0;
        virtual bool SetVRWWWL(float fWW, float fWL);
        virtual bool SetVRWWWL(float fWW, float fWL, unsigned char nLabel);
        virtual bool SetObjectAlpha(float fAlpha);
        virtual bool SetObjectAlpha(float fAlpha, unsigned char nLabel);
        virtual bool SetTransferFunc(std::map<int, RGBA> ctrlPts);
        virtual bool SetTransferFunc(std::map<int, RGBA> ctrlPts, unsigned char nLabel);
        virtual bool SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts);
        virtual bool SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts, unsigned char nLabel);

        // cpr
        virtual bool SetCPRLinePatient(std::vector<Point3d> cprLine);
        virtual bool SetCPRLineVoxel(std::vector<Point3d> cprLine);
        virtual std::vector<Point3d> GetCPRLineVoxel();
        virtual bool RotateCPR(float angle, PlaneType planeType);

    protected:
        DataManager m_dataMan;
    };

}


