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

#include "MPRInfo.h"
#include "DataManager.h"
#include "Methods.h"

using namespace MonkeyGL;

MPRInfo::MPRInfo()
{
    m_pDataManager = NULL;
}

MPRInfo::~MPRInfo()
{
}

bool MPRInfo::SetDataManager(DataManager* pDataManager)
{
    if (NULL == pDataManager){
        return false;
    }
    m_pDataManager = pDataManager;
    return true;
}


bool MPRInfo::GetPlaneInitSize(int& nWidth, int& nHeight, int& nNumber, int dim[], double spacing[], const PlaneType& planeType)
{
    if (dim[0]<=0 || dim[1]<=0 || dim[2]<=0)
		return false;
	if (spacing[0]<=0 || spacing[1]<=0 || spacing[2]<=0)
		return false;

	double minSpacing = spacing[0]<spacing[1] ? spacing[0]:spacing[1];
	minSpacing = minSpacing<spacing[2] ? minSpacing:spacing[2];	

	switch (planeType)
	{
	case PlaneAxial:
	case PlaneAxialOblique:
		{
			nWidth = dim[0]*spacing[0]/minSpacing;
			nHeight = dim[1]*spacing[1]/minSpacing;
			nNumber = dim[2]*spacing[2]/minSpacing;
			return true;
		}
		break;
	case PlaneSagittal:
	case PlaneSagittalOblique:
		{
			nWidth = dim[1]*spacing[1]/minSpacing;
			nHeight = dim[2]*spacing[2]/minSpacing;
			nNumber = dim[0]*spacing[0]/minSpacing;
			return true;
		}
		break;
	case PlaneCoronal:
	case PlaneCoronalOblique:
		{
			nWidth = dim[0]*spacing[0]/minSpacing;
			nHeight = dim[2]*spacing[2]/minSpacing;
			nNumber = dim[1]*spacing[1]/minSpacing;
			return true;
		}
		break;
	case PlaneVR:
		{
			nWidth = 512;
			nHeight = 512;
			nNumber = 1;
			return true;
		}
		break;
	case PlaneNotDefined:
	default:
		{
			nWidth = -1;
			nHeight = -1;
			nNumber = -1;
		}
		break;
	}
	return false;
}

bool MPRInfo::ResetPlaneInfos()
{
    if (NULL == m_pDataManager){
        return false;
    }
    int dim[3] = {m_pDataManager->GetDim(0), m_pDataManager->GetDim(1), m_pDataManager->GetDim(2)};
    double spacing[3] = {m_pDataManager->GetSpacing(0), m_pDataManager->GetSpacing(1), m_pDataManager->GetSpacing(2)};
	m_ptCrossHair = Point3d(0.5*dim[0]*spacing[0], 
							0.5*dim[1]*spacing[1], 
							0.5*dim[2]*spacing[2]);

	m_planeInfos.clear();
	for (int i = (int)PlaneAxial; i<=(int)PlaneCoronalOblique; i++)
	{
		PlaneType planeType = (PlaneType)i;
		PlaneInfo info;
		info.m_PlaneType = planeType;
		GetPlaneInitSize(info.m_nWidth, info.m_nHeight, info.m_nNumber, dim, spacing, planeType);
		Direction3d& dirH = info.m_dirH;
		Direction3d& dirV = info.m_dirV;
		switch (planeType)
		{
		case PlaneAxial:
			{
				dirH = Direction3d(1, 0, 0);
				dirV = Direction3d(0, 1, 0);
			}
			break;
		case PlaneAxialOblique:
			{
				dirH = Direction3d(1, 0, 0);
				dirV = Direction3d(0, 1, 0);
			}
			break;
		case PlaneSagittal:
		case PlaneSagittalOblique:
			{
				dirH = Direction3d(0, 1, 0);
				dirV = Direction3d(0, 0, -1);
			}
			break;
		case PlaneCoronal:
		case PlaneCoronalOblique:
			{
				dirH = Direction3d(1, 0, 0);
				dirV = Direction3d(0, 0, -1);
			}
			break;
		case PlaneNotDefined:
			break;
		default:
			break;
		}
		info.m_fPixelSpacing = m_pDataManager->GetMinSpacing();
		info.m_fSliceThickness = m_pDataManager->GetMinSpacing();
		m_planeInfos[planeType] = info;

		m_ptCenter = m_ptCrossHair;
	}
    return true;
}

bool MPRInfo::IsExistGroupPlaneInfos( PlaneType planeType )
{
	switch (planeType)
	{
		break;
	case PlaneAxial:
	case PlaneSagittal:
	case PlaneCoronal:
		{
			return (m_planeInfos.find(PlaneAxial) != m_planeInfos.end()) &&
				(m_planeInfos.find(PlaneSagittal) != m_planeInfos.end() ) &&
				(m_planeInfos.find(PlaneCoronal) != m_planeInfos.end());
		}
		break;
	case PlaneAxialOblique:
	case PlaneSagittalOblique:
	case PlaneCoronalOblique:
		{
			return (m_planeInfos.find(PlaneAxialOblique) != m_planeInfos.end()) &&
				(m_planeInfos.find(PlaneSagittalOblique) != m_planeInfos.end() ) &&
				(m_planeInfos.find(PlaneCoronalOblique) != m_planeInfos.end());
		}
		break;
	case PlaneNotDefined:
		break;
	default:
		break;
	}
	return false;
}

std::vector<PlaneType> MPRInfo::GetCrossPlaneType( PlaneType planeType )
{
	std::vector<PlaneType> vecPlaneTypes;
	switch (planeType)
	{
	case PlaneAxial:
		{
			vecPlaneTypes.push_back(PlaneSagittal);
			vecPlaneTypes.push_back(PlaneCoronal);
		}
		break;
	case PlaneSagittal:
		{
			vecPlaneTypes.push_back(PlaneAxial);
			vecPlaneTypes.push_back(PlaneCoronal);
		}
		break;
	case PlaneCoronal:
		{
			vecPlaneTypes.push_back(PlaneAxial);
			vecPlaneTypes.push_back(PlaneSagittal);
		}
		break;
	case PlaneAxialOblique:
		{
			vecPlaneTypes.push_back(PlaneSagittalOblique);
			vecPlaneTypes.push_back(PlaneCoronalOblique);
		}
		break;
	case PlaneSagittalOblique:
		{
			vecPlaneTypes.push_back(PlaneAxialOblique);
			vecPlaneTypes.push_back(PlaneCoronalOblique);
		}
		break;
	case PlaneCoronalOblique:
		{
			vecPlaneTypes.push_back(PlaneAxialOblique);
			vecPlaneTypes.push_back(PlaneSagittalOblique);
		}
		break;
	case PlaneNotDefined:
		break;
	default:
		break;
	}
	return vecPlaneTypes;
}

void MPRInfo::SetMPRType( MPRType type )
{
	for (std::map<PlaneType, PlaneInfo>::iterator iter = m_planeInfos.begin();
		iter != m_planeInfos.end(); ++iter)
	{
		iter->second.m_MPRType = type;
	}
}

bool MPRInfo::GetPlaneInfo(PlaneType planeType, PlaneInfo& info){
    if (m_planeInfos.find(planeType) == m_planeInfos.end())
        return false;
    info = m_planeInfos[planeType];
    return true;
}

void MPRInfo::UpdateThickness( double val )
{
	for (std::map<PlaneType, PlaneInfo>::iterator iter = m_planeInfos.begin();
		iter != m_planeInfos.end(); ++iter)
	{
		iter->second.m_fSliceThickness = val;
	}
}

void MPRInfo::SetThickness(double val, PlaneType planeType)
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return;
	m_planeInfos[planeType].m_fSliceThickness = val;
}

bool MPRInfo::GetThickness(double& val, PlaneType planeType)
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return false;
	val = m_planeInfos[planeType].m_fSliceThickness;
	return true;
}

double MPRInfo::GetPixelSpacing( PlaneType planeType )
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return 1.0;
	return m_planeInfos[planeType].m_fPixelSpacing;
}

bool MPRInfo::GetPlaneSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
    PlaneInfo info;
    if (!GetPlaneInfo(planeType, info))
        return false;

    nWidth = m_planeInfos[planeType].m_nWidth;
    nHeight = m_planeInfos[planeType].m_nHeight;
    return true;
}

bool MPRInfo::GetPlaneNumber( int& nNumber, const PlaneType& planeType )
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return false;

	nNumber = m_planeInfos[planeType].m_nNumber;
	return true;
}

bool MPRInfo::GetPlaneIndex( int& index, const PlaneType& planeType )
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return false;
	int nTotalNumber = m_planeInfos[planeType].m_nNumber;
	double spacing = m_pDataManager->GetMinSpacing();
	double distCrossHair2Center = Methods::Distance_Point2Plane(m_ptCrossHair, info.m_dirH, info.m_dirV, m_ptCenter);
	int nDeltaNum = distCrossHair2Center/spacing;
	index = nDeltaNum + (nTotalNumber-1)/2;
	return true;
}

void MPRInfo::SetPlaneIndex( int index, PlaneType planeType )
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return;
	PlaneInfo& info = m_planeInfos[planeType];
	Direction3d dirN = info.GetNormDirection();
	int nTotalNum = info.m_nNumber;
	double spacing = m_pDataManager->GetMinSpacing();
	double dist2Center = (index - (nTotalNum-1)/2) * spacing;
	Point3d ptProj = Methods::Projection_Point2Plane(m_ptCrossHair, info.m_dirH, info.m_dirV, m_ptCenter);
	m_ptCrossHair = ptProj + dirN*dist2Center;
}

bool MPRInfo::GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType )
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return false;
	Direction3d dirZ = info.m_dirH.cross(info.m_dirV);
	pMatirx[0] = info.m_dirH.x();
	pMatirx[1] = info.m_dirH.y();
	pMatirx[2] = info.m_dirH.z();
	pMatirx[3] = info.m_dirV.x();
	pMatirx[4] = info.m_dirV.y();
	pMatirx[5] = info.m_dirV.z();
	pMatirx[6] = dirZ.x();
	pMatirx[7] = dirZ.y();
	pMatirx[8] = dirZ.z();
	return true;
}

void MPRInfo::Browse( float fDelta, PlaneType planeType )
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return;
	PlaneInfo& info = m_planeInfos[planeType];
	Direction3d dirN = info.GetNormDirection();
	m_ptCrossHair = m_ptCrossHair + dirN*fDelta;
}
void MPRInfo::PanCrossHair(float fx, float fy, PlaneType planeType)
{
	Point3d ptVoxel;
	if (!TransferImage2Voxel(ptVoxel, fx, fy, planeType))
		return;

	m_ptCrossHair = ptVoxel;
}

void MPRInfo::RotateCrossHair( float fAngle, PlaneType planeType )
{
	Direction3d dirA = m_planeInfos[planeType].GetNormDirection();

	fAngle = fAngle/180*PI;
	double x = dirA.x();
	double y = dirA.y();
	double z = dirA.z();
	double xx = x*x;
	double yy = y*y;
	double zz = z*z;
	double sinV = sin(fAngle);
	double cosV = cos(fAngle);
	double cosVT = 1.0 - cosV;
	double m[3][3];
	m[0][0] = xx + (1-xx)*cosV;
	m[1][1] = yy + (1-yy)*cosV;
	m[2][2] = zz + (1-zz)*cosV;
	m[0][1] = x*y*cosVT - z*sinV;
	m[0][2] = x*z*cosVT + y*sinV;
	m[1][0] = x*y*cosVT + z*sinV;
	m[1][2] = y*z*cosVT - x*sinV;
	m[2][0] = x*z*cosVT - y*sinV;
	m[2][1] = y*z*cosVT + x*sinV;

	std::vector<PlaneType> vecCrossPlaneTypes = GetCrossPlaneType(planeType);
	for (size_t i=0; i<vecCrossPlaneTypes.size(); i++)
	{
		PlaneType crossPlaneType = vecCrossPlaneTypes[i];
		
		Direction3d& dirH = m_planeInfos[crossPlaneType].m_dirH;
		Direction3d& dirV = m_planeInfos[crossPlaneType].m_dirV;
		Direction3d dirN = m_planeInfos[crossPlaneType].GetNormDirection();
		Point3d ptCenter = GetCenterPointPlane(dirN);
		double fW = m_planeInfos[crossPlaneType].m_fPixelSpacing*m_planeInfos[crossPlaneType].m_nWidth;
		double fH = m_planeInfos[crossPlaneType].m_fPixelSpacing*m_planeInfos[crossPlaneType].m_nHeight;
		Point3d ptLeftTop = ptCenter - dirH*(fW/2) - dirV*(fH/2);
		Point3d ptRightTop = ptCenter + dirH*(fW/2) - dirV*(fH/2);
		Point3d ptLeftBottom = ptCenter - dirH*(fW/2) + dirV*(fH/2);

		ptCenter -= m_ptCrossHair;
		ptCenter = Methods::GetTransferPoint(m, ptCenter);
		ptCenter += m_ptCrossHair;

		ptLeftTop -= m_ptCrossHair;
		ptLeftTop = Methods::GetTransferPoint(m, ptLeftTop);
		ptLeftTop += m_ptCrossHair;
		ptRightTop -= m_ptCrossHair;
		ptRightTop = Methods::GetTransferPoint(m, ptRightTop);
		ptRightTop += m_ptCrossHair;
		ptLeftBottom -= m_ptCrossHair;
		ptLeftBottom = Methods::GetTransferPoint(m, ptLeftBottom);
		ptLeftBottom += m_ptCrossHair;

		dirH = Direction3d(ptRightTop[0]-ptLeftTop[0], ptRightTop[1]-ptLeftTop[1], ptRightTop[2]-ptLeftTop[2]);
		dirV = Direction3d(ptLeftBottom[0]-ptLeftTop[0], ptLeftBottom[1]-ptLeftTop[1], ptLeftBottom[2]-ptLeftTop[2]);

		UpdatePlaneSize(crossPlaneType);
	}
}

void MPRInfo::UpdatePlaneSize(PlaneType planeType)
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return;
	double xLen = m_pDataManager->GetDim(0) * m_pDataManager->GetSpacing(0);
	double yLen = m_pDataManager->GetDim(1) * m_pDataManager->GetSpacing(1);
	double zLen = m_pDataManager->GetDim(2) * m_pDataManager->GetSpacing(2);
	double spacing = m_pDataManager->GetMinSpacing();

	int nWidth = 0, nHeight = 0, nNumber = 0;
	if (1)
	{
		std::vector<Point3d> ptVertexes = m_pDataManager->GetVertexes();

		double xNegative = 0, xPositive = 0;
		double yNegative = 0, yPositive = 0;
		double zNegative = 0, zPositive = 0;
		for (auto i=0; i<ptVertexes.size(); i++)
		{
			Point3d ptProj = Methods::Projection_Point2Plane(ptVertexes[i], info.m_dirH, info.m_dirV, m_ptCrossHair);

			double x = Methods::Length_VectorInLine(ptProj, info.m_dirH, m_ptCrossHair);
			if (x >= 0)
			{
				xPositive = xPositive > x ? xPositive : x;
			}
			else
			{
				xNegative = xNegative < x ? xNegative : x;
			}

			double y = Methods::Length_VectorInLine(ptProj, info.m_dirV, m_ptCrossHair);
			if (y >= 0)
			{
				yPositive = yPositive > y ? yPositive : y;
			}
			else
			{
				yNegative = yNegative < y ? yNegative : y;
			}

			double z = Methods::Distance_Point2Plane(ptVertexes[i], info.m_dirH, info.m_dirV, m_ptCrossHair);
			if (z >= 0)
			{
				zPositive = zPositive > z ? zPositive : z;
			}
			else
			{
				zNegative = zNegative < z ? zNegative : z;
			}
		}

		nWidth = (xPositive - xNegative)/spacing;
		nHeight = (yPositive - yNegative)/spacing;
		nNumber = (zPositive - zNegative)/spacing;
	}
	else
	{
		nWidth = Methods::GetLengthofCrossLineInBox(info.m_dirH, spacing, 0, xLen, 0, yLen, 0, zLen);
		nHeight = Methods::GetLengthofCrossLineInBox(info.m_dirV, spacing, 0, xLen, 0, yLen, 0, zLen);
	}

	m_planeInfos[planeType].m_nWidth = nWidth;
	m_planeInfos[planeType].m_nHeight = nHeight;
	m_planeInfos[planeType].m_nNumber = nNumber;
}

bool MPRInfo::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return false;

	PlaneInfo& info = m_planeInfos[planeType];
	Direction3d dirN = info.GetNormDirection();
	Point3d vec = m_ptCrossHair - GetCenterPointPlane(dirN);
	double lenH = vec.x()*info.m_dirH.x() + vec.y()*info.m_dirH.y() + vec.z()*info.m_dirH.z();
	double lenV = vec.x()*info.m_dirV.x() + vec.y()*info.m_dirV.y() + vec.z()*info.m_dirV.z();
	x = (info.m_nWidth-1)/2.0 + lenH/info.m_fPixelSpacing;
	y = (info.m_nHeight-1)/2.0 + lenV/info.m_fPixelSpacing;

	return true;
}

bool MPRInfo::TransferImage2Voxel(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType)
{
	Point3d ptObject;
	if (!TransferImage2Voxel(ptObject, xImage, yImage, planeType))
		return false;

	x = ptObject[0];
	y = ptObject[1];
	z = ptObject[2];
	return true;
}

bool MPRInfo::TransferImage2Voxel(Point3d& ptVoxel, double xImage, double yImage, PlaneType planeType)
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return false;

	PlaneInfo& info = m_planeInfos[planeType];
	Point3d ptCenter = GetCenterPointPlane(info.GetNormDirection());
	Point3d ptLeftTop = info.GetLeftTopPoint(ptCenter);
	ptVoxel = ptLeftTop + info.m_dirH*(xImage*info.m_fPixelSpacing);
	ptVoxel = ptVoxel + info.m_dirV*(yImage*info.m_fPixelSpacing);

	return true;
}

bool MPRInfo::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
{
	if (!IsExistGroupPlaneInfos(planeType))
		return false;

	switch (planeType)
	{
	case PlaneAxial:
	case PlaneSagittal:
	case PlaneCoronal:
		{
			dirH = Direction2d(1,0);
			dirV = Direction2d(0,1);
		}
		break;
	case PlaneAxialOblique:
		{
			Direction3d dir3HSelf = m_planeInfos[planeType].m_dirH;
			Direction3d dir3VSelf = m_planeInfos[planeType].m_dirV;

			Direction3d dir3H = m_planeInfos[PlaneSagittalOblique].GetNormDirection();
			Direction3d dir3V = m_planeInfos[PlaneCoronalOblique].GetNormDirection();

			double lenH = dir3H.x()*dir3HSelf.x() + dir3H.y()*dir3HSelf.y() + dir3H.z()*dir3HSelf.z();
			double lenV = dir3H.x()*dir3VSelf.x() + dir3H.y()*dir3VSelf.y() + dir3H.z()*dir3VSelf.z();
			dirH = Direction2d(lenH, lenV);

			lenH = dir3V.x()*dir3HSelf.x() + dir3V.y()*dir3HSelf.y() + dir3V.z()*dir3HSelf.z();
			lenV = dir3V.x()*dir3VSelf.x() + dir3V.y()*dir3VSelf.y() + dir3V.z()*dir3VSelf.z();
			dirV = Direction2d(lenH, lenV);
		}
		break;
	case PlaneSagittalOblique:
		{
			Direction3d dir3HSelf = m_planeInfos[planeType].m_dirH;
			Direction3d dir3VSelf = m_planeInfos[planeType].m_dirV;

			Direction3d dir3H = m_planeInfos[PlaneCoronalOblique].GetNormDirection();
			Direction3d dir3V = m_planeInfos[PlaneAxialOblique].GetNormDirection();

			double lenH = dir3H.x()*dir3HSelf.x() + dir3H.y()*dir3HSelf.y() + dir3H.z()*dir3HSelf.z();
			double lenV = dir3H.x()*dir3VSelf.x() + dir3H.y()*dir3VSelf.y() + dir3H.z()*dir3VSelf.z();
			dirH = Direction2d(lenH, lenV);

			lenH = dir3V.x()*dir3HSelf.x() + dir3V.y()*dir3HSelf.y() + dir3V.z()*dir3HSelf.z();
			lenV = dir3V.x()*dir3VSelf.x() + dir3V.y()*dir3VSelf.y() + dir3V.z()*dir3VSelf.z();
			dirV = Direction2d(lenH, lenV);
		}
		break;
	case PlaneCoronalOblique:
		{
			Direction3d dir3HSelf = m_planeInfos[planeType].m_dirH;
			Direction3d dir3VSelf = m_planeInfos[planeType].m_dirV;

			Direction3d dir3H = m_planeInfos[PlaneSagittalOblique].GetNormDirection();
			Direction3d dir3V = m_planeInfos[PlaneAxialOblique].GetNormDirection();

			double lenH = dir3H.x()*dir3HSelf.x() + dir3H.y()*dir3HSelf.y() + dir3H.z()*dir3HSelf.z();
			double lenV = dir3H.x()*dir3VSelf.x() + dir3H.y()*dir3VSelf.y() + dir3H.z()*dir3VSelf.z();
			dirH = Direction2d(lenH, lenV);

			lenH = dir3V.x()*dir3HSelf.x() + dir3V.y()*dir3HSelf.y() + dir3V.z()*dir3HSelf.z();
			lenV = dir3V.x()*dir3VSelf.x() + dir3V.y()*dir3VSelf.y() + dir3V.z()*dir3VSelf.z();
			dirV = Direction2d(lenH, lenV);
		}
		break;

	default:
		break;
	}

	return true;
}

bool MPRInfo::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	if (m_planeInfos.find(planeType) == m_planeInfos.end())
		return false;
	dir3dH = m_planeInfos[planeType].m_dirH;
	dir3dV = m_planeInfos[planeType].m_dirV;
	return true;
}

Point3d MPRInfo::GetCenterPointPlane(Direction3d dirN){
	return Methods::Projection_Point2Plane(m_ptCenter, dirN, m_ptCrossHair);
}

Point3d MPRInfo::GetCrossHair()
{
    return m_ptCrossHair;
}

void MPRInfo::SetCrossHair(Point3d pt)
{
    m_ptCrossHair = pt;
}

Point3d MPRInfo::GetCenterPoint()
{
    return m_ptCenter;
}