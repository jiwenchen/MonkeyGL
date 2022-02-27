#pragma once
#include "Direction.h"
#include "Point.h"
#include "PlaneDefines.h"

namespace MonkeyGL{

	enum MPRType
	{
		eMPRType_Average = 0,
		eMPRType_MIP,
		eMPRType_MinIP
	};

	class PlaneInfo
	{
	public:
		PlaneInfo(void);
		~PlaneInfo(void);

	public:
		Direction3d GetNormDirection(){
			return m_dirH.cross(m_dirV);
		}

		Point3d GetLeftTopPoint(Point3d ptCenter){
			Point3d ptLeftTop = ptCenter - m_dirH*(0.5*m_nWidth*m_fPixelSpacing);
			ptLeftTop = ptLeftTop - m_dirV*(0.5*m_nHeight*m_fPixelSpacing);
			return ptLeftTop;
		}

	public:
		ePlaneType m_PlaneType;
		Direction3d m_dirH;
		Direction3d m_dirV;
		int m_nWidth;
		int m_nHeight;
		int m_nNumber;
		double m_fPixelSpacing;
		double m_fSliceThickness;
		MPRType m_MPRType;
	};

}
