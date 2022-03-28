#pragma once
#include <map>
#include "Defines.h"
#include "VolumeInfo.h"
#include "TransferFunction.h"
#include "PlaneDefines.h"
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
        virtual void SetVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        virtual void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        virtual void SetAnisotropy(double x, double y, double z);
        virtual void Reset();
        virtual void SetTransferFunc(const std::map<int, RGBA>& ctrlPoints);
        virtual void SetTransferFunc(const std::map<int, RGBA>& rgbPoints, const std::map<int, double>& alphaPoints);

        virtual void Browse(float fDelta, ePlaneType planeType);
        virtual void SetPlaneIndex(int index, ePlaneType planeType);
        virtual void PanCrossHair(int nx, int ny, ePlaneType planeType);
        virtual void RotateCrossHair(float fAngle, ePlaneType planeType);
        virtual void UpdateThickness(double val);
        virtual void SetThickness(double val, ePlaneType planeType);
        virtual bool GetThickness(double& val, ePlaneType planeType);
        virtual void SetMPRType(MPRType type);

    // output
        virtual short* GetVolumeData();
        virtual bool GetPlaneMaxSize(int& nWidth, int& nHeight, const ePlaneType& planeType);
        virtual bool GetPlaneData(short* pData, int& nWidth, int& nHeight, const ePlaneType& planeType);

        virtual bool GetCrossHairPoint(double& x, double& y, const ePlaneType& planeType) = 0;
        virtual bool TransferImage2Object(double& x, double& y, double& z, double xImage, double yImage, ePlaneType planeType);
        virtual bool GetCrossHairPoint3D(Point3d& pt);
        virtual bool GetDirection(Direction2d& dirH, Direction2d& dirV, const ePlaneType& planeType);
        virtual bool GetDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, const ePlaneType& planeType);
        virtual bool GetBatchDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const ePlaneType& planeType);

        virtual double GetPixelSpacing(ePlaneType planeType);

        virtual bool GetVRData(unsigned char* pVR, int nWidth, int nHeight) = 0;
        virtual void SaveVR2BMP(const char* szFile, int nWidth, int nHeight) = 0;

        virtual bool GetBatchData( std::vector<short*>& vecBatchData, BatchInfo batchInfo ) = 0;

        virtual bool GetPlaneIndex(int& index, ePlaneType planeType);
        virtual bool GetPlaneNumber(int& nTotalNum, ePlaneType planeType);
        virtual bool GetPlaneRotateMatrix(float* pMatirx, ePlaneType planeType);


        virtual void Anterior() = 0;
        virtual void Posterior() = 0;
        virtual void Left() = 0;
        virtual void Right() = 0;
        virtual void Head() = 0;
        virtual void Foot() = 0;

        virtual void Rotate(float fxRotate, float fyRotate) = 0;
        virtual void Zoom(float ratio) = 0;
        virtual void Pan(float fxShift, float fyShift) = 0;
        virtual void SetWL(float fWW, float fWL) = 0;

    protected:
        DataManager m_dataMan;
    };

}


