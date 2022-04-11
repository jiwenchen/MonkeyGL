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
#include "TransferFunction.h"
#include "VolumeInfo.h"
#include "Defines.h"
#include "PlaneInfo.h"
#include <vector>

namespace MonkeyGL {

    class DataManager
    {
    public:
        DataManager(void);
        ~DataManager(void);

    public:
        void SetControlPoints_TF(std::map<int,RGBA> ctrlPts);
        void SetControlPoints_TF(std::map<int,RGBA> rgbPts, std::map<int, double> alphaPts);
        bool GetTransferFunction(RGBA*& pBuffer, int& nLen);

        void SetColorBackground(float clrBkg[]);
        float* GetColorBackground();

        Orientation& GetOrientation(){
            return m_orientation;
        }

        bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);
        bool LoadVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        void SetAnisotropy(double x, double y, double z);
        void Reset();
        std::shared_ptr<short> GetVolumeData();
        std::shared_ptr<short> GetVolumeData(int& nWidth, int& nHeight, int& nDepth);
        int GetDim(int index);
        double GetAnisotropy(int index);
        double GetMinAnisotropy();
        bool GetPlaneMaxSize(int& nWidth, int& nHeight, const PlaneType& planeType);
        bool GetPlaneSize(int& nWidth, int& nHeight, const PlaneType& planeType);
        bool GetPlaneNumber(int& nNumber, const PlaneType& planeType);
        bool GetPlaneIndex(int& index, const PlaneType& planeType);
        bool GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType );

        void Browse(float fDelta, PlaneType planeType);
        void SetPlaneIndex( int index, PlaneType planeType );
        void PanCrossHair(int nx, int ny, PlaneType planeType);
        void RotateCrossHair( float fAngle, PlaneType planeType );
        void UpdateThickness(double val);
        void SetThickness(double val, PlaneType planeType);
        bool GetThickness(double& val, PlaneType planeType);
        void SetMPRType(MPRType type);
        Point3d GetCrossHair(){
            return m_ptCrossHair;
        }
        void SetCrossHair(Point3d pt){
            m_ptCrossHair = pt;
        }
        Point3d GetCenterPoint(){
            return m_ptCenter;
        }
        Point3d GetCrossHair_Voxel(){
            return Point3d(m_ptCrossHair.x()/m_volInfo.GetAnisotropy(0),
                m_ptCrossHair.y()/m_volInfo.GetAnisotropy(1),
                m_ptCrossHair.z()/m_volInfo.GetAnisotropy(2));
        }
        bool GetCrossHairPoint(double& x, double& y, const PlaneType& planeType);
        bool TransferImage2Object(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType);
        bool TransferImage2Object(Point3d& ptObject, double xImage, double yImage, PlaneType planeType);
        bool GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType );
        bool GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType );
        bool GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType );

        bool GetPlaneInfo(PlaneType planeType, PlaneInfo& info){
            if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
                return false;
            info = m_mapPlaneType2Info[planeType];
            return true;
        }
        Point3d Object2Voxel(Point3d ptObject){
            return Point3d(ptObject.x()/m_volInfo.GetAnisotropy(0),
                ptObject.y()/m_volInfo.GetAnisotropy(1),
                ptObject.z()/m_volInfo.GetAnisotropy(2));
        }
        Point3d GetCenterPointPlane(Direction3d dirN){
            return DataManager::GetProjectPoint(dirN, m_ptCrossHair, m_ptCenter);
        }

        double GetPixelSpacing(PlaneType planeType);

        static Point3d GetProjectPoint(Direction3d dirN, Point3d ptPlane, Point3d ptNeed2Project);

        static int TrimValue(int nValue, int nMin, int nMax){
            nValue = nValue>=nMin ? nValue:nMin;
            nValue = nValue<=nMax ? nValue:nMin;
            return nValue;
        }

    private:
        void ResetPlaneInfos();
        bool IsExistGroupPlaneInfos(PlaneType planeType);
        PlaneType GetHorizonalPlaneType(PlaneType planeType);
        PlaneType GetVerticalPlaneType(PlaneType planeType);

        Point3d GetTransferPoint(double m[3][3], Point3d pt);
        std::vector<PlaneType> GetCrossPlaneType(PlaneType planeType);
        void UpdatePlaneSize(PlaneType planeType);

        std::vector<Point3d> GetVertexes();

    private:
        TransferFunction m_tf;
        VolumeInfo m_volInfo;
        Orientation m_orientation;
        Point3d m_ptCrossHair;
        Point3d m_ptCenter;
        float m_colorBkg[4];

        std::map<PlaneType, PlaneInfo> m_mapPlaneType2Info;

        bool m_bHaveVolumeInfo;
        bool m_bHaveAnisotropy;
    };

}