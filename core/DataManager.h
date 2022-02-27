#pragma once
#include "TransferFunction.h"
#include "VolumeInfo.h"
#include "Defines.h"
#include "PlaneDefines.h"
#include "PlaneInfo.h"
#include <vector>

namespace MonkeyGL {

    class DataManager
    {
    public:
        DataManager(void);
        ~DataManager(void);

    public:
        void SetMinPos_TF(int pos);
        void SetMaxPos_TF(int pos);
        void SetControlPoints_TF(std::map<int,RGBA> ctrlPts);
        void SetControlPoints_TF(std::map<int,RGBA> rgbPts, std::map<int, double> alphaPts);
        bool GetTransferFunction(RGBA*& pBuffer, int& nLen);


        ORIENTATION& GetOrientation(){
            return m_orientation;
        }

        bool LoadVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        void SetAnisotropy(double x, double y, double z);
        void Reset();
        short* GetVolumeData();
        int GetDim(int index);
        double GetAnisotropy(int index);
        double GetMinAnisotropy();
        bool GetPlaneMaxSize(int& nWidth, int& nHeight, const ePlaneType& planeType);
        bool GetPlaneSize(int& nWidth, int& nHeight, const ePlaneType& planeType);
        bool GetPlaneNumber(int& nNumber, const ePlaneType& planeType);
        bool GetPlaneIndex(int& index, const ePlaneType& planeType);
        bool GetPlaneRotateMatrix( float* pMatirx, ePlaneType planeType );

        void Browse(float fDelta, ePlaneType planeType);
        void SetPlaneIndex( int index, ePlaneType planeType );
        void PanCrossHair(int nx, int ny, ePlaneType planeType);
        void RotateCrossHair( float fAngle, ePlaneType planeType );
        void UpdateThickness(double val);
        void SetThickness(double val, ePlaneType planeType);
        bool GetThickness(double& val, ePlaneType planeType);
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
        bool GetCrossHairPoint(double& x, double& y, const ePlaneType& planeType);
        bool TransferImage2Object(double& x, double& y, double& z, double xImage, double yImage, ePlaneType planeType);
        bool TransferImage2Object(Point3d& ptObject, double xImage, double yImage, ePlaneType planeType);
        bool GetDirection( Direction2d& dirH, Direction2d& dirV, const ePlaneType& planeType );
        bool GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const ePlaneType& planeType );
        bool GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const ePlaneType& planeType );

        bool GetPlaneInfo(ePlaneType planeType, PlaneInfo& info){
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

        double GetPixelSpacing(ePlaneType planeType);

        static Point3d GetProjectPoint(Direction3d dirN, Point3d ptPlane, Point3d ptNeed2Project);

        static int TrimValue(int nValue, int nMin, int nMax){
            nValue = nValue>=nMin ? nValue:nMin;
            nValue = nValue<=nMax ? nValue:nMin;
            return nValue;
        }

    private:
        void ResetPlaneInfos();
        bool IsExistGroupPlaneInfos(ePlaneType planeType);
        ePlaneType GetHorizonalPlaneType(ePlaneType planeType);
        ePlaneType GetVerticalPlaneType(ePlaneType planeType);

        Point3d GetTransferPoint(double m[3][3], Point3d pt);
        std::vector<ePlaneType> GetCrossPlaneType(ePlaneType planeType);
        void UpdatePlaneSize(ePlaneType planeType);

        std::vector<Point3d> GetVertexes();

    private:
        TransferFunction m_tf;
        VolumeInfo m_volInfo;
        ORIENTATION m_orientation;
        Point3d m_ptCrossHair;
        Point3d m_ptCenter;

        std::map<ePlaneType, PlaneInfo> m_mapPlaneType2Info;

        bool m_bHaveVolumeInfo;
        bool m_bHaveAnisotropy;
    };

}