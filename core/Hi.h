#pragma once
#include "Defines.h"
#include <map>
#include <vector>
#include "PlaneDefines.h"
#include "Direction.h"
#include "PlaneInfo.h"
#include "BatchInfo.h"

namespace MonkeyGL {

    class Hi
    {
    public:
        Hi();
        ~Hi(void);

    public:
        void SetVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        void SetAnisotropy(double x, double y, double z);
        void Reset();
        void SetTransferFunc(const std::map<int, RGBA>& ctrlPoints);
        void SetTransferFunc(const std::map<int, RGBA>& rgbPoints, const std::map<int, double>& alphaPoints);

 
        void Browse(float fDelta, ePlaneType planeType);	
        void PanCrossHair(int nx, int ny, ePlaneType planeType);
        void RotateCrossHair(float fAngle, ePlaneType planeType);
        void SetPlaneIndex(int index, ePlaneType planeType);

    // output
        bool GetPlaneMaxSize(int& nWidth, int& nHeight, const ePlaneType& planeType);
        bool GetPlaneData(short* pData, int& nWidth, int& nHeight, const ePlaneType& planeType);

        bool GetCrossHairPoint(double& x, double& y, const ePlaneType& planeType);
        bool TransferImage2Object(double& x, double& y, double& z, double xImage, double yImage, ePlaneType planeType);
        bool GetCrossHairPoint3D(Point3d& pt);
        bool GetDirection(Direction2d& dirH, Direction2d& dirV, const ePlaneType& planeType);
        bool GetDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, const ePlaneType& planeType);
        bool GetBatchDirection3D(Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const ePlaneType& planeType);

        double GetPixelSpacing(ePlaneType planeType);

        bool GetVRData(unsigned char* pVR, int nWidth, int nHeight);

        bool GetBatchData(std::vector<short*>& vecBatchData, const BatchInfo& batchInfo);

        bool GetPlaneIndex(int& index, ePlaneType planeType);
        bool GetPlaneNumber(int& nTotalNum, ePlaneType planeType);
        
        bool GetPlaneRotateMatrix(float* pMatrix, ePlaneType planeType);

        void Anterior();
        void Posterior();
        void Left();
        void Right();
        void Head();
        void Foot();

        void Rotate(float fxRotate, float fyRotate);
        void Zoom(float ratio);
        void Pan(float fxShift, float fyShift);
        void SetWL(float fWW, float fWL);

        void UpdateThickness(double val);
        void SetThickness(double val, ePlaneType planeType);
        bool GetThickness(double& val, ePlaneType planeType);
        void SetMPRType(MPRType type);
    };

}
