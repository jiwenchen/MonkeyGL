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
#include <vector>
#include <set>
#include "TransferFunction.h"
#include "VolumeInfo.h"
#include "Defines.h"
#include "PlaneInfo.h"
#include "ObjectInfo.h"
#include "MPRInfo.h"
#include "CPRInfo.h"

namespace MonkeyGL {

    class DataManager
    {
    public:
        DataManager(void);
        ~DataManager(void);

    public:
        bool LoadVolumeFile(const char* szFile);
        bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);
        unsigned char AddObjectMaskFile(const char* szFile);
        unsigned char AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth);
        bool UpdateActiveObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth);
        bool UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel);
        std::shared_ptr<short> GetVolumeData();
        std::shared_ptr<short> GetVolumeData(int& nWidth, int& nHeight, int& nDepth);
        std::shared_ptr<unsigned char> GetMaskData();

        void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        void SetSpacing(double x, double y, double z);
        void SetOrigin(Point3d pt);

        bool SetVRWWWL(float fWW, float fWL);
        bool SetVRWWWL(float fWW, float fWL, unsigned char nLabel);
        bool SetObjectAlpha(float fAlpha);
        bool SetObjectAlpha(float fAlpha, unsigned char nLabel);
        bool SetControlPoints_TF(std::map<int, RGBA> ctrlPts);
        bool SetControlPoints_TF(std::map<int, RGBA> ctrlPts, unsigned char nLabel);
        bool SetControlPoints_TF(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts);
        bool SetControlPoints_TF(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts, unsigned char nLabel);
        bool LoadTransferFunction(const char* szFile);
        bool SaveTransferFunction(const char* szFile);
        std::map<unsigned char, ObjectInfo> GetObjectInfos();

        void Reset();

        void SetColorBackground(RGBA clrBG);
        RGBA GetColorBackground();

        bool Need2InvertZ(){
            return m_volInfo.Need2InvertZ();
        }

        int GetDim(int index);
        double GetSpacing(int index);
        double GetMinSpacing();
        bool GetPlaneMaxSize(int& nWidth, int& nHeight, const PlaneType& planeType);

        bool GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType );
        std::vector<Point3d> GetVertexes();

        // MPR
        void SetMPRType(MPRType type);
        bool GetPlaneInfo(PlaneType planeType, PlaneInfo& info);
        void UpdateThickness(double val);
        void SetThickness(double val, PlaneType planeType);
        double GetThickness(PlaneType planeType);
        double GetPixelSpacing(PlaneType planeType);
        bool GetPlaneSize(int& nWidth, int& nHeight, const PlaneType& planeType);
        bool GetPlaneNumber(int& nNumber, const PlaneType& planeType);
        bool GetPlaneIndex(int& index, const PlaneType& planeType);
        void SetPlaneIndex( int index, PlaneType planeType );
        bool GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType );
        void Browse(float fDelta, PlaneType planeType);
        void PanCrossHair(float fx, float fy, PlaneType planeType);
        void RotateCrossHair( float fAngle, PlaneType planeType );
        bool GetCrossHairPoint(double& x, double& y, const PlaneType& planeType);
        bool TransferImage2Voxel(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType);
        bool TransferImage2Voxel(Point3d& ptVoxel, double xImage, double yImage, PlaneType planeType);
        bool GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType );
        bool GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType );
        Point3d GetCenterPointPlane(Direction3d dirN);
        Point3d GetCrossHair();
        void SetCrossHair(Point3d pt);
        Point3d GetCenterPoint();

        //cpr
        bool SetCPRLinePatient(std::vector<Point3d> cprLine);
        bool SetCPRLineVoxel(std::vector<Point3d> cprLine);
        std::vector<Point3d> GetCPRLineVoxel();
        bool RotateCPR(float angle, PlaneType planeType);
        bool GetCPRInfo(Point3d*& pPoints, Direction3d*& pDirs, int& nWidth, int& nHeight, PlaneType planeType);

        Point3d Patient2Voxel(Point3d ptPatient){
            return m_volInfo.Patient2Voxel(ptPatient);
        }
        Point3d Voxel2Patient(Point3d ptVoxel){
            return m_volInfo.Voxel2Patient(ptVoxel);
        }

    private:
        void ClearAndReset();

    private:
        VolumeInfo m_volInfo;
        MPRInfo m_mprInfo;
        CPRInfo m_cprInfo;

        int m_activeLabel;
        std::map<unsigned char, ObjectInfo> m_objectInfos;

        RGBA m_colorBG;
    };

}