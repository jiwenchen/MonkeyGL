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

#include "CPRInfo.h"
#include "Methods.h"
#include "DataManager.h"
#include "Logger.h"

using namespace MonkeyGL;

CPRInfo::CPRInfo()
{
	m_pDataManager = NULL;
	m_dirStretchedCPR = Direction3d(0, 0, 1);
	m_angleStrechedCPR = 0;
    m_angleStraightenedCPR = 0;
    m_spacing = 1.0;
}

CPRInfo::~CPRInfo()
{
}

void CPRInfo::SetDataManager(DataManager* pDataManager)
{
	m_pDataManager = pDataManager;
}

void CPRInfo::SetSpacing(float spacing)
{
    m_spacing = spacing;
}

bool CPRInfo::RotateCPR(float angle, PlaneType planeType)
{
    if (PlaneStretchedCPR == planeType){
        m_angleStrechedCPR += angle;
		return true;
    }
    else if (PlaneStraightenedCPR == planeType){
        m_angleStraightenedCPR += angle;
        return true;
    }
    
    return false;
}

bool CPRInfo::GetCPRInfo(Point3d*& pPoints, Direction3d*& pDirs, int& len, PlaneType planeType)
{
	len = m_cprLineVoxel.size();
	if (len < 2){
		return false;
	}
	if (PlaneStretchedCPR == planeType){
		return GetCPRInfoStretched(pPoints, pDirs, len);
	}
	else if (PlaneStraightenedCPR == planeType){
		return GetCPRInfoStraightened(pPoints, pDirs, len);
	}
	return false;
}

bool CPRInfo::GetCPRInfoStretched(Point3d*& pPoints, Direction3d*& pDirs, int& len)
{
	Point3d ptStart = m_cprLineVoxel.front();
	Point3d ptRotate = Methods::RotatePoint3D(m_ptOriginStrechedCPR, m_dirStretchedCPR, ptStart, m_angleStrechedCPR*PI/180.0);
    Direction3d dir = Direction3d(ptRotate, ptStart);
	pPoints = new Point3d[len];
	pDirs = new Direction3d[len];
	
	pPoints[0] = Methods::Projection_Point2Plane(ptStart, dir, ptRotate);
	pDirs[0] = dir;
	int idx = 0;
	Point3d ptPre = pPoints[0];
	Point3d ptCur;
	for(size_t i=1; i<len; i++){
		ptCur = Methods::Projection_Point2Plane(m_cprLineVoxel[i], dir, ptRotate);
		if (ptCur.DistanceTo(ptPre) >= m_spacing){
			ptPre = ptCur;
			idx++;
			pPoints[idx] = ptPre;
			pDirs[idx] = dir;
		}
	}
	return true;
}

bool CPRInfo::GetCPRInfoStraightened(Point3d*& pPoints, Direction3d*& pDirs, int& len)
{
	len = m_cprLineVoxel.size();
	pPoints = new Point3d[len];
	pDirs = new Direction3d[len];
	
	Point3d ptStart = m_cprLineVoxel.front();
	Direction3d dirPath(ptStart, m_cprLineVoxel[1]);
	Point3d ptRotate = Methods::RotatePoint3D(m_ptOriginStrechedCPR, dirPath, ptStart, m_angleStrechedCPR*PI/180.0);
	Direction3d dir(ptRotate, ptStart);
	pPoints[0] = ptRotate;
	pDirs[0] = dir;
	double r = ptRotate.DistanceTo(ptStart);

	for (int i=1; i<m_cprLineVoxel.size()-1; i++){
		Point3d ptPre = m_cprLineVoxel[i-1];
		Point3d ptCur = m_cprLineVoxel[i];
		Point3d ptNxt = m_cprLineVoxel[i+1];

		Direction3d dir1 = pDirs[i-1];
		Direction3d dir2(ptPre, ptCur);
		Direction3d dir3(ptCur, ptNxt);

		Direction3d dirTemp = dir1.cross(dir2);
		Direction3d dir = dir3.cross(dirTemp);
		if (dir.Length() < 0.99f)
			dir = dir1;
		if (dir.dot(dir1) < 0)
			dir = dir.negative();

		pDirs[i] = dir;

		pPoints[i] = ptCur - pDirs[i]*r;
	}
	return true;
}

bool CPRInfo::SetCPRLinePatient(std::vector<Point3d> cprLine)
{
	if (NULL == m_pDataManager)
	{
		return false;
	}
	m_cprLineVoxel.clear();
	for (size_t i=0; i<cprLine.size(); i++){
		m_cprLineVoxel.push_back(m_pDataManager->Patient2Voxel(cprLine[i]));
	}
	return UpdateCPRInfo();
}

bool CPRInfo::SetCPRLineVoxel(std::vector<Point3d> cprLine)
{
	m_cprLineVoxel = cprLine;
	return UpdateCPRInfo();
}

std::vector<Point3d> CPRInfo::GetCPRLineVoxel()
{
	return m_cprLineVoxel;
}

Direction3d CPRInfo::FirstDirectionProjection(Point3d pt, Direction3d dirN)
{
	double x, y, z;
	if (abs(dirN.y()) > 0.3)
	{
		x = pt.x();
		y = pt.y() - dirN.z()/dirN.y();
		z = pt.z() + 1;
	}
	else if (abs(dirN.x()) > 0.3) 
	{
		x = pt.x() - dirN.z()/dirN.x();
		y = pt.y();
		z = pt.z() + 1;
	}
	else// if (abs(dirN.z()) > 0.3) 
	{
		x = pt.x() + 1;
		y = pt.y();
		z = pt.z() - dirN.x()/dirN.z();
	}
	return Direction3d(pt, Point3d(x, y, z));
}

bool CPRInfo::UpdateCPRInfo()
{
	if (m_cprLineVoxel.size() < 2){
		return false;
	}

	Point3d ptStart = m_cprLineVoxel.front();
	Point3d ptEnd = m_cprLineVoxel.back();

	m_dirStretchedCPR = Direction3d(ptStart, ptEnd);

	m_Radius = 0;
	for (int i=1; i<m_cprLineVoxel.size(); i++){
		double r = Methods::Distance_Point2Line(m_cprLineVoxel[i], m_dirStretchedCPR, ptStart);
		if (r > m_Radius)
			m_Radius = r;
	}
	m_Radius = 50;

	Direction3d dirStart = CPRInfo::FirstDirectionProjection(ptStart, m_dirStretchedCPR);
	m_ptOriginStrechedCPR = ptStart + dirStart * GetRadius();
	m_angleStrechedCPR = 0;
	m_angleStraightenedCPR = 0;
	return true;
}

bool CPRInfo::GetPlaneSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	if (m_cprLineVoxel.size() <= 0)
		return false;
    nWidth = GetRadius()*2.0 + 1;
	if (nWidth%2 == 1)
	{
		nWidth += 1;
	}
    nHeight = m_cprLineVoxel.size();
    return true;
}