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

#include "DataManager.h"
#include <vector>

using namespace MonkeyGL;

DataManager::DataManager(void)
{
	m_orientation.rx = 1.0f;
	m_orientation.ry = 0.0f;
	m_orientation.rz = 0.0f;
	m_orientation.cx = 0.0f;
	m_orientation.cy = 1.0f;
	m_orientation.cz = 0.0f;

	memset(m_colorBkg, 0, 4*sizeof(float));

	m_bHaveVolumeInfo = false;
	m_bHaveAnisotropy = false;
}

DataManager::~DataManager(void)
{
}

void DataManager::SetControlPoints_TF( std::map<int,RGBA> ctrlPts )
{
	m_tf.SetControlPoints(ctrlPts);
}

void DataManager::SetControlPoints_TF( std::map<int,RGBA> rgbPts, std::map<int, double> alphaPts )
{
	m_tf.SetControlPoints(rgbPts, alphaPts);
}

bool DataManager::GetTransferFunction( RGBA*& pBuffer, int& nLen )
{
	return m_tf.GetTransferFunction(pBuffer, nLen);
}

void DataManager::SetColorBackground(float clrBkg[])
{
	memcpy(m_colorBkg, clrBkg, 4*sizeof(float));
}

float* DataManager::GetColorBackground()
{
	return m_colorBkg;
}

bool DataManager::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	m_bHaveVolumeInfo =  m_volInfo.SetVolumeData(pData, nWidth, nHeight, nDepth);
	
	ResetPlaneInfos();
	return m_bHaveVolumeInfo;
}

bool DataManager::LoadVolumeFile( const char* szFile, int nWidth, int nHeight, int nDepth )
{
	m_bHaveVolumeInfo = m_volInfo.LoadVolumeFile(szFile, nWidth, nHeight, nDepth);

	ResetPlaneInfos();
	return m_bHaveVolumeInfo;
}

void DataManager::SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ)
{
	m_volInfo.SetDirection(dirX, dirY, dirZ);
}

void DataManager::SetAnisotropy( double x, double y, double z )
{
	m_volInfo.SetAnisotropy(x, y, z);
	m_bHaveAnisotropy = true;
	ResetPlaneInfos();
}

void DataManager::Reset()
{
	ResetPlaneInfos();
}

void DataManager::ResetPlaneInfos()
{
	if (!m_bHaveVolumeInfo || !m_bHaveAnisotropy)
		return;

	m_ptCrossHair = Point3d(0.5*m_volInfo.GetDim(0)*m_volInfo.GetAnisotropy(0), 
							0.5*m_volInfo.GetDim(1)*m_volInfo.GetAnisotropy(1), 
							0.5*m_volInfo.GetDim(2)*m_volInfo.GetAnisotropy(2));

	m_mapPlaneType2Info.clear();
	for (int i = (int)PlaneAxial; i<=(int)PlaneCoronalOblique; i++)
	{
		PlaneType pt = (PlaneType)i;
		PlaneInfo info;
		info.m_PlaneType = pt;
		m_volInfo.GetPlaneInitSize(info.m_nWidth, info.m_nHeight, info.m_nNumber, pt);
		Direction3d& dirH = info.m_dirH;
		Direction3d& dirV = info.m_dirV;
		switch (pt)
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
		info.m_fPixelSpacing = m_volInfo.GetMinAnisotropy();
		info.m_fSliceThickness = m_volInfo.GetMinAnisotropy();
		m_mapPlaneType2Info[pt] = info;

		m_ptCenter = m_ptCrossHair;
	}
}

std::shared_ptr<short> DataManager::GetVolumeData()
{
	return m_volInfo.GetVolumeData();
}

std::shared_ptr<short> DataManager::GetVolumeData(int& nWidth, int& nHeight, int& nDepth)
{
	return m_volInfo.GetVolumeData(nWidth, nHeight, nDepth);
}

int DataManager::GetDim( int index )
{
	return m_volInfo.GetDim(index);
}

double DataManager::GetAnisotropy(int index)
{
	return m_volInfo.GetAnisotropy(index);
}

double DataManager::GetMinAnisotropy()
{
	return m_volInfo.GetMinAnisotropy();
}

double Distance_Point2Line(Point3d pt, Direction3d dirNorm, Point3d ptLine)
{
	Point3d vec = pt - ptLine;
	return vec.x()*dirNorm.x() + vec.y()*dirNorm.y() + vec.z()*dirNorm.z();
}

double Distance_Point2Plane(Point3d pt, Direction3d dirH, Direction3d dirV, Point3d ptPlane)
{
	Direction3d dirNorm = dirH.cross(dirV);
	Point3d vec = pt - ptPlane;
	return vec.x()*dirNorm.x() + vec.y()*dirNorm.y() + vec.z()*dirNorm.z();
}

Point3d Projection_Point2Plane(Point3d pt, Direction3d dirH, Direction3d dirV, Point3d ptPlane)
{
	Direction3d dirNorm = dirH.cross(dirV);
	double dist = Distance_Point2Plane(pt, dirH, dirV, ptPlane);
	return Point3d(pt.x()-dist*dirNorm.x(), pt.y()-dist*dirNorm.y(), pt.z()-dist*dirNorm.z());
}

int GetLengthofCrossLineInBox(Direction3d dir, double ans, double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
{
	double xCenter = (xMin + xMax) / 2;
	double yCenter = (yMin + yMax) / 2;
	double zCenter = (zMin + zMax) / 2;

	int nUpPart = 0, nDownPart = 0;
	for (int i=0; ;i++)
	{
		double xTemp = xCenter + i*ans*dir.x();
		double yTemp = yCenter + i*ans*dir.y();
		double zTemp = zCenter + i*ans*dir.z();
		if (xTemp<=xMin || xTemp>=xMax || yTemp<=yMin || yTemp>=yMax || zTemp<=zMin || zTemp>=zMax)
		{
			nUpPart = i;
			break;
		}
	}
	for (int i=0; ;i--)
	{
		double xTemp = xCenter + i*ans*dir.x();
		double yTemp = yCenter + i*ans*dir.y();
		double zTemp = zCenter + i*ans*dir.z();
		if (xTemp<=xMin || xTemp>=xMax || yTemp<=yMin || yTemp>=yMax || zTemp<=zMin || zTemp>=zMax)
		{
			nDownPart = -i;
			break;
		}
	}
	return nUpPart + nDownPart;
}

void DataManager::UpdatePlaneSize(PlaneType planeType)
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return;
	double xLen = GetDim(0) * GetAnisotropy(0);
	double yLen = GetDim(1) * GetAnisotropy(1);
	double zLen = GetDim(2) * GetAnisotropy(2);
	double ans = GetMinAnisotropy();

	int nWidth=0, nHeight = 0, nNumber = 0;
	if (1)
	{
		std::vector<Point3d> ptVertexes = GetVertexes();

		double xNegative = 0, xPositive = 0;
		double yNegative = 0, yPositive = 0;
		double zNegative = 0, zPositive = 0;
		for (auto i=0; i<ptVertexes.size(); i++)
		{
			Point3d ptProj = Projection_Point2Plane(ptVertexes[i], info.m_dirH, info.m_dirV, m_ptCrossHair);

			double x = Distance_Point2Line(ptProj, info.m_dirH, m_ptCrossHair);
			if (x >= 0)
			{
				xPositive = xPositive > x ? xPositive : x;
			}
			else
			{
				xNegative = xNegative < x ? xNegative : x;
			}

			double y = Distance_Point2Line(ptProj, info.m_dirV, m_ptCrossHair);
			if (y >= 0)
			{
				yPositive = yPositive > y ? yPositive : y;
			}
			else
			{
				yNegative = yNegative < y ? yNegative : y;
			}

			double z = Distance_Point2Plane(ptVertexes[i], info.m_dirH, info.m_dirV, m_ptCrossHair);
			if (z >= 0)
			{
				zPositive = zPositive > z ? zPositive : z;
			}
			else
			{
				zNegative = zNegative < z ? zNegative : z;
			}
		}

		nWidth = (xPositive - xNegative)/ans;
		nHeight = (yPositive - yNegative)/ans;
		nNumber = (zPositive - zNegative)/ans;
	}
	else
	{
		nWidth = GetLengthofCrossLineInBox(info.m_dirH, ans, 0, xLen, 0, yLen, 0, zLen);
		nHeight = GetLengthofCrossLineInBox(info.m_dirV, ans, 0, xLen, 0, yLen, 0, zLen);
	}

	m_mapPlaneType2Info[planeType].m_nWidth = nWidth;
	m_mapPlaneType2Info[planeType].m_nHeight = nHeight;
	m_mapPlaneType2Info[planeType].m_nNumber = nNumber;
}

std::vector<Point3d> DataManager::GetVertexes()
{
	double xLen = GetDim(0) * GetAnisotropy(0);
	double yLen = GetDim(1) * GetAnisotropy(1);
	double zLen = GetDim(2) * GetAnisotropy(2);
	double ans = GetMinAnisotropy();
	std::vector<Point3d> ptVertexes;
	ptVertexes.push_back(Point3d(0, 0, 0));
	ptVertexes.push_back(Point3d(xLen, 0, 0));
	ptVertexes.push_back(Point3d(0, yLen, 0));
	ptVertexes.push_back(Point3d(xLen, yLen, 0));
	ptVertexes.push_back(Point3d(0, 0, zLen));
	ptVertexes.push_back(Point3d(xLen, 0, zLen));
	ptVertexes.push_back(Point3d(0, yLen, zLen));
	ptVertexes.push_back(Point3d(xLen, yLen, zLen));
	return ptVertexes;
}

bool DataManager::GetPlaneSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return false;

	nWidth = m_mapPlaneType2Info[planeType].m_nWidth;
	nHeight = m_mapPlaneType2Info[planeType].m_nHeight;
	return true;
}

bool DataManager::GetPlaneNumber( int& nNumber, const PlaneType& planeType )
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return false;

	nNumber = m_mapPlaneType2Info[planeType].m_nNumber;
	return true;
}

bool DataManager::GetPlaneIndex( int& index, const PlaneType& planeType )
{
	PlaneInfo info;
	if (!GetPlaneInfo(planeType, info))
		return false;
	int nTotalNumber = m_mapPlaneType2Info[planeType].m_nNumber;
	double ans = GetMinAnisotropy();
	double distCrossHair2Center = Distance_Point2Plane(m_ptCrossHair, info.m_dirH, info.m_dirV, m_ptCenter);
	int nDeltaNum = distCrossHair2Center/ans;
	index = nDeltaNum + (nTotalNumber-1)/2;
	return true;
}

bool DataManager::GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType )
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

bool DataManager::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	if (planeType == PlaneVR)
	{
		nWidth = 512;
		nHeight = 512;
		return true;
	}
	else
	{
		PlaneInfo info;
		if (!GetPlaneInfo(planeType, info))
			return false;
		double xLen = GetDim(0) * GetAnisotropy(0);
		double yLen = GetDim(1) * GetAnisotropy(1);
		double zLen = GetDim(2) * GetAnisotropy(2);
		double ans = GetMinAnisotropy();
		nWidth = int(sqrtf(xLen*xLen + yLen*yLen + zLen*zLen)/ans) + 1;
		nHeight = nWidth;
		return true;
	}
}

void DataManager::Browse( float fDelta, PlaneType planeType )
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return;
	PlaneInfo& info = m_mapPlaneType2Info[planeType];
	Direction3d dirN = info.GetNormDirection();
	m_ptCrossHair = m_ptCrossHair + dirN*fDelta;
}

void DataManager::SetPlaneIndex( int index, PlaneType planeType )
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return;
	PlaneInfo& info = m_mapPlaneType2Info[planeType];
	Direction3d dirN = info.GetNormDirection();
	int nTotalNum = info.m_nNumber;
	double ans = GetMinAnisotropy();
	double dist2Center = (index - (nTotalNum-1)/2) * ans;
	Point3d ptProj = Projection_Point2Plane(m_ptCrossHair, info.m_dirH, info.m_dirV, m_ptCenter);
	m_ptCrossHair = ptProj + dirN*dist2Center;
}

void DataManager::PanCrossHair(int nx, int ny, PlaneType planeType)
{
	Point3d ptObject;
	if (!TransferImage2Object(ptObject, nx, ny, planeType))
		return;

	m_ptCrossHair = ptObject;
}

void  DataManager::RotateCrossHair( float fAngle, PlaneType planeType )
{
	Direction3d dirA = m_mapPlaneType2Info[planeType].GetNormDirection();

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
		
		Direction3d& dirH = m_mapPlaneType2Info[crossPlaneType].m_dirH;
		Direction3d& dirV = m_mapPlaneType2Info[crossPlaneType].m_dirV;
		Direction3d dirN = m_mapPlaneType2Info[crossPlaneType].GetNormDirection();
		Point3d ptCenter = GetCenterPointPlane(dirN);
		double fW = m_mapPlaneType2Info[crossPlaneType].m_fPixelSpacing*m_mapPlaneType2Info[crossPlaneType].m_nWidth;
		double fH = m_mapPlaneType2Info[crossPlaneType].m_fPixelSpacing*m_mapPlaneType2Info[crossPlaneType].m_nHeight;
		Point3d ptLeftTop = ptCenter - dirH*(fW/2) - dirV*(fH/2);
		Point3d ptRightTop = ptCenter + dirH*(fW/2) - dirV*(fH/2);
		Point3d ptLeftBottom = ptCenter - dirH*(fW/2) + dirV*(fH/2);

		ptCenter -= m_ptCrossHair;
		ptCenter = GetTransferPoint(m, ptCenter);
		ptCenter += m_ptCrossHair;

		ptLeftTop -= m_ptCrossHair;
		ptLeftTop = GetTransferPoint(m, ptLeftTop);
		ptLeftTop += m_ptCrossHair;
		ptRightTop -= m_ptCrossHair;
		ptRightTop = GetTransferPoint(m, ptRightTop);
		ptRightTop += m_ptCrossHair;
		ptLeftBottom -= m_ptCrossHair;
		ptLeftBottom = GetTransferPoint(m, ptLeftBottom);
		ptLeftBottom += m_ptCrossHair;

		dirH = Direction3d(ptRightTop[0]-ptLeftTop[0], ptRightTop[1]-ptLeftTop[1], ptRightTop[2]-ptLeftTop[2]);
		dirV = Direction3d(ptLeftBottom[0]-ptLeftTop[0], ptLeftBottom[1]-ptLeftTop[1], ptLeftBottom[2]-ptLeftTop[2]);

		UpdatePlaneSize(crossPlaneType);
	}
}

Point3d DataManager::GetTransferPoint(double m[3][3], Point3d pt)
{
	double r[3];
	for (int i=0; i<3; i++)
	{
		r[i] = 0;
		for (int j=0; j<3; j++)
		{
			r[i] += m[i][j]*pt[j];
		}
	}
	return Point3d(r[0], r[1], r[2]);
}

bool DataManager::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return false;

	PlaneInfo& info = m_mapPlaneType2Info[planeType];
	Direction3d dirN = info.GetNormDirection();
	Point3d vec = m_ptCrossHair - GetCenterPointPlane(dirN);
	double lenH = vec.x()*info.m_dirH.x() + vec.y()*info.m_dirH.y() + vec.z()*info.m_dirH.z();
	double lenV = vec.x()*info.m_dirV.x() + vec.y()*info.m_dirV.y() + vec.z()*info.m_dirV.z();
	x = (info.m_nWidth-1)/2.0 + lenH/info.m_fPixelSpacing;
	y = (info.m_nHeight-1)/2.0 + lenV/info.m_fPixelSpacing;

	return true;
}

bool DataManager::TransferImage2Object(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType)
{
	Point3d ptObject;
	if (!TransferImage2Object(ptObject, xImage, yImage, planeType))
		return false;

	x = ptObject[0];
	y = ptObject[1];
	z = ptObject[2];
	return true;
}

bool DataManager::TransferImage2Object(Point3d& ptObject, double xImage, double yImage, PlaneType planeType)
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return false;

	PlaneInfo& info = m_mapPlaneType2Info[planeType];
	Point3d ptCenter = GetCenterPointPlane(info.GetNormDirection());
	Point3d ptLeftTop = info.GetLeftTopPoint(ptCenter);
	ptObject = ptLeftTop + info.m_dirH*(xImage*info.m_fPixelSpacing);
	ptObject = ptObject + info.m_dirV*(yImage*info.m_fPixelSpacing);

	return true;
}

bool DataManager::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
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
			Direction3d dir3HSelf = m_mapPlaneType2Info[planeType].m_dirH;
			Direction3d dir3VSelf = m_mapPlaneType2Info[planeType].m_dirV;

			Direction3d dir3H = m_mapPlaneType2Info[PlaneSagittalOblique].GetNormDirection();
			Direction3d dir3V = m_mapPlaneType2Info[PlaneCoronalOblique].GetNormDirection();

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
			Direction3d dir3HSelf = m_mapPlaneType2Info[planeType].m_dirH;
			Direction3d dir3VSelf = m_mapPlaneType2Info[planeType].m_dirV;

			Direction3d dir3H = m_mapPlaneType2Info[PlaneCoronalOblique].GetNormDirection();
			Direction3d dir3V = m_mapPlaneType2Info[PlaneAxialOblique].GetNormDirection();

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
			Direction3d dir3HSelf = m_mapPlaneType2Info[planeType].m_dirH;
			Direction3d dir3VSelf = m_mapPlaneType2Info[planeType].m_dirV;

			Direction3d dir3H = m_mapPlaneType2Info[PlaneSagittalOblique].GetNormDirection();
			Direction3d dir3V = m_mapPlaneType2Info[PlaneAxialOblique].GetNormDirection();

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

bool DataManager::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return false;
	dir3dH = m_mapPlaneType2Info[planeType].m_dirH;
	dir3dV = m_mapPlaneType2Info[planeType].m_dirV;
	return true;
}

bool DataManager::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType )
{
	Direction3d dirH_Plane, dirV_Plane;
	GetDirection3D(dirH_Plane, dirV_Plane, planeType);
	Direction3d dirA = dirH_Plane.cross(dirV_Plane);

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

	Point3d pt = GetTransferPoint(m, Point3d(dirH_Plane.x(), dirH_Plane.y(), dirH_Plane.z()));
	dir3dH = Direction3d(pt.x(), pt.y(), pt.z());
	dir3dV = dirA;
	return true;
}

bool DataManager::IsExistGroupPlaneInfos( PlaneType planeType )
{
	switch (planeType)
	{
		break;
	case PlaneAxial:
	case PlaneSagittal:
	case PlaneCoronal:
		{
			return (m_mapPlaneType2Info.find(PlaneAxial) != m_mapPlaneType2Info.end()) &&
				(m_mapPlaneType2Info.find(PlaneSagittal) != m_mapPlaneType2Info.end() ) &&
				(m_mapPlaneType2Info.find(PlaneCoronal) != m_mapPlaneType2Info.end());
		}
		break;
	case PlaneAxialOblique:
	case PlaneSagittalOblique:
	case PlaneCoronalOblique:
		{
			return (m_mapPlaneType2Info.find(PlaneAxialOblique) != m_mapPlaneType2Info.end()) &&
				(m_mapPlaneType2Info.find(PlaneSagittalOblique) != m_mapPlaneType2Info.end() ) &&
				(m_mapPlaneType2Info.find(PlaneCoronalOblique) != m_mapPlaneType2Info.end());
		}
		break;
	case PlaneNotDefined:
		break;
	default:
		break;
	}
	return false;
}

PlaneType DataManager::GetHorizonalPlaneType( PlaneType planeType )
{
	switch (planeType)
	{
	case PlaneAxial:
		return PlaneCoronal;
		break;
	case PlaneSagittal:
		break;
	case PlaneCoronal:
		break;
	case PlaneAxialOblique:
		break;
	case PlaneSagittalOblique:
		break;
	case PlaneCoronalOblique:
		break;
	case PlaneNotDefined:
		break;
	default:
		break;
	}

	return PlaneNotDefined;
}

PlaneType DataManager::GetVerticalPlaneType( PlaneType planeType )
{
	return PlaneNotDefined;
}

std::vector<PlaneType> DataManager::GetCrossPlaneType( PlaneType planeType )
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

Point3d DataManager::GetProjectPoint( Direction3d dirN, Point3d ptPlane, Point3d ptNeed2Project )
{
	Point3d vec = ptPlane - ptNeed2Project;
	double len = dirN.x()*vec.x() + dirN.y()*vec.y() + dirN.z()*vec.z();
	return (ptNeed2Project + dirN*len);
}

double DataManager::GetPixelSpacing( PlaneType planeType )
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return 1.0;
	return m_mapPlaneType2Info[planeType].m_fPixelSpacing;
}

void DataManager::UpdateThickness( double val )
{
	for (std::map<PlaneType, PlaneInfo>::iterator iter = m_mapPlaneType2Info.begin();
		iter != m_mapPlaneType2Info.end(); ++iter)
	{
		iter->second.m_fSliceThickness = val;
	}
}

void DataManager::SetThickness(double val, PlaneType planeType)
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return;
	m_mapPlaneType2Info[planeType].m_fSliceThickness = val;
}

bool DataManager::GetThickness(double& val, PlaneType planeType)
{
	if (m_mapPlaneType2Info.find(planeType) == m_mapPlaneType2Info.end())
		return false;
	val = m_mapPlaneType2Info[planeType].m_fSliceThickness;
	return true;
}

void DataManager::SetMPRType( MPRType type )
{
	for (std::map<PlaneType, PlaneInfo>::iterator iter = m_mapPlaneType2Info.begin();
		iter != m_mapPlaneType2Info.end(); ++iter)
	{
		iter->second.m_MPRType = type;
	}
}
