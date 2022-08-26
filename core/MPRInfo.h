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
#include <memory>
#include <map>
#include "Point.h"
#include "Direction.h"
#include "Defines.h"
#include "PlaneInfo.h"

namespace MonkeyGL
{
    class DataManager;

    class MPRInfo
    {
    public:
        MPRInfo();
        ~MPRInfo();

    public:
        bool SetDataManager(DataManager* pDataManager);
        void SetMPRType(MPRType type);
        bool GetPlaneInfo(PlaneType planeType, PlaneInfo& info);
        void UpdateThickness(double val);
        void SetThickness(double val, PlaneType planeType);
        bool GetThickness(double& val, PlaneType planeType);
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

        bool ResetPlaneInfos();

    private:
        bool GetPlaneInitSize(int& nWidth, int& nHeight, int& nNumber, int dim[], double spacing[], const PlaneType& planeType);
        bool IsExistGroupPlaneInfos(PlaneType planeType);
        std::vector<PlaneType> GetCrossPlaneType(PlaneType planeType);
        void UpdatePlaneSize(PlaneType planeType);

    private:
        DataManager* m_pDataManager;
        std::map<PlaneType, PlaneInfo> m_planeInfos;
        Point3d m_ptCrossHair;
        Point3d m_ptCenter;
    };
    
}
