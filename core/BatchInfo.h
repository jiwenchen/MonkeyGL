#pragma once
#include "Direction.h"
#include "PlaneInfo.h"

namespace MonkeyGL{

    class BatchInfo
    {
    public:
        BatchInfo(){
            memset(this, 0, sizeof(BatchInfo));
        }

    public:
        Direction3d m_dirH;
        Direction3d m_dirV;
        Point3d m_ptCenter;
        double m_fLengthH;
        double m_fLengthV;
        double m_fPixelSpacing;
        double m_fSliceThickness;
        double m_fSliceDistance;
        int m_nNum;
        MPRType m_MPRType;

        int Width(){
            return int(m_fLengthH / m_fPixelSpacing);
        }
        int Height(){
            return int(m_fLengthV / m_fPixelSpacing);
        }
    };

}
