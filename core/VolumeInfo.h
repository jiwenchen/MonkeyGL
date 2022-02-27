#pragma once
#include "Point.h"
#include "Direction.h"
#include <string>
#include "PlaneDefines.h"

namespace MonkeyGL {

    class VolumeInfo
    {
    public:
        VolumeInfo(void);
        ~VolumeInfo(void);

    public:
        void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        bool LoadVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        int GetVolumeSize(){
            return m_Dims[0]*m_Dims[1]*m_Dims[2];
        }
        int GetVolumeBytes(){
            return GetVolumeSize()*sizeof(short);
        }
        void SetAnisotropy(double x, double y, double z){
            m_Anisotropy[0] = x;
            m_Anisotropy[1] = y;
            m_Anisotropy[2] = z;
        }
        void SetSliceThickness(double sliceTh){
            m_fSliceThickness = sliceTh;
        }

        short* GetVolumeData(){
            return m_pVolume;
        }

        int GetDim(int index){
            return m_Dims[index];
        }
        double GetAnisotropy(int index){
            return m_Anisotropy[index];
        }
        double GetMinAnisotropy()
        {
            double ani = m_Anisotropy[0]<m_Anisotropy[1] ? m_Anisotropy[0] : m_Anisotropy[1];
            return ani<m_Anisotropy[2] ? ani : m_Anisotropy[2];
        }

        bool GetPlaneInitSize(int& nWidth, int& nHeight, int& nNumber, const ePlaneType& planeType);

        bool IsInvertZ();
        bool IsPerpendicularCoord();
        void NormVolumeData();

    private:
        short* m_pVolume;
        double m_fSliceThickness; //mm
        double m_fSlope;
        double m_fIntercept;
        int m_Dims[3];
        double m_Anisotropy[3];
        Point3d m_ptStart;
        Direction3d m_dirX;
        Direction3d m_dirY;
        Direction3d m_dirZ;
    };

}