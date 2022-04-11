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

#include "VolumeInfo.h"
#include "Defines.h"
#include "DataManager.h"
#include <cstring>
#include "StopWatch.h"

using namespace MonkeyGL;

VolumeInfo::VolumeInfo( void )
{
	m_pVolume.reset();
	m_fSliceThickness = 1.0;
	m_fSlope = 1;
	m_fIntercept = 0;
	memset(m_Dims, 0, 3*sizeof(int));
	memset(m_Anisotropy, 0, 3*sizeof(double));
	m_dirX = Direction3d(1, 0, 0);
	m_dirY = Direction3d(0, 1, 0);
	m_dirZ = Direction3d(0, 0, 1);
}

VolumeInfo::~VolumeInfo( void )
{
}

void VolumeInfo::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	m_dirX = dirX;
	m_dirY = dirY;
	m_dirZ = dirZ;
}

bool VolumeInfo::IsInvertZ()
{
	Direction3d dirNorm = m_dirX.cross(m_dirY);
	if (dirNorm.dot(m_dirZ) > 0)
		return true;
	return false;
}

bool VolumeInfo::IsPerpendicularCoord()
{
	Direction3d dirNorm = m_dirX.cross(m_dirY);
	double dotValue = abs(dirNorm.dot(m_dirZ));
	double vRef = sinf(PI*88/180);
	return dotValue >= vRef;
}

void VolumeInfo::NormVolumeData()
{
	if (NULL == m_pVolume.get())
		return;

	if (!IsInvertZ())
	{
		int nSizeSlice = m_Dims[0] * m_Dims[1];
		short* pslice = new short[nSizeSlice];
		for (int i=0; i<m_Dims[2]/2; i++)
		{
			memcpy(pslice, m_pVolume.get() + nSizeSlice * i, nSizeSlice * sizeof(short));
			memcpy(m_pVolume.get() + nSizeSlice * i, m_pVolume.get() + nSizeSlice * (m_Dims[2]-1-i), nSizeSlice * sizeof(short));
			memcpy(m_pVolume.get() + nSizeSlice * (m_Dims[2] - 1 - i), pslice, nSizeSlice * sizeof(short));
		}
		delete[] pslice;

		m_dirZ = Direction3d(-m_dirZ.x(), -m_dirZ.y(), -m_dirZ.z());
	}

	if (IsPerpendicularCoord())
		return;

	Direction3d dirNorm = m_dirX.cross(m_dirY);
	double xLen = m_Anisotropy[0] * m_Dims[0];
	double yLen = m_Anisotropy[1] * m_Dims[1];
	double zLen = m_Anisotropy[2] * m_Dims[2];

	double cosV = abs(dirNorm.dot(m_dirZ));
	if (cosV == 0)
	{
		return;
	}

	Point3d ptLeftTop_StartSlice(0, 0, 0);
	Point3d ptLeftTop_EndSlice = ptLeftTop_StartSlice + Point3d(m_dirZ.x()*zLen/cosV, m_dirZ.y()*zLen/cosV, m_dirZ.z()*zLen/cosV);
	Point3d ptProj_LeftTop_EndSlice = DataManager::GetProjectPoint(dirNorm, ptLeftTop_StartSlice, ptLeftTop_EndSlice);

	Direction3d dirShift_proj(
		ptProj_LeftTop_EndSlice.x() - ptLeftTop_StartSlice.x(),
		ptProj_LeftTop_EndSlice.y() - ptLeftTop_StartSlice.y(),
		ptProj_LeftTop_EndSlice.z() - ptLeftTop_StartSlice.z()
		);

	Point3d ptShift_proj = ptProj_LeftTop_EndSlice - ptLeftTop_StartSlice;

	Point3d ptStart = ptLeftTop_StartSlice;
	if (dirShift_proj.dot(m_dirX) < 0)
	{
		ptStart.SetX(ptProj_LeftTop_EndSlice.x());
	}
	if (dirShift_proj.dot(m_dirY) < 0)
	{
		ptStart.SetY(ptProj_LeftTop_EndSlice.y());
	}
	xLen += ptShift_proj.x();
	yLen += ptShift_proj.y();

	int nWidth = xLen / m_Anisotropy[0];
	int nHeight = yLen / m_Anisotropy[1];

	//m_Dims[0] = nWidth;
	//m_Dims[1] = nHeight;
	std::shared_ptr<short> pVolumeExt(new short[nWidth*nHeight*m_Dims[2]]);
	for (auto i=0; i<m_Dims[2]; i++)
	{
		double zdelta = i*m_Anisotropy[2];
		Point3d ptLT(ptStart.x()+zdelta*dirNorm.x(),
			ptStart.y()+zdelta*dirNorm.y(),
			ptStart.z()+zdelta*dirNorm.z()
			);

		Point3d ptLT_ori(m_dirZ.x()*zdelta/cosV,
			m_dirZ.y()*zdelta/cosV,
			m_dirZ.z()*zdelta/cosV
			);

		double xdelta = abs(ptLT_ori.x() - ptLT.x());
		double ydelta = abs(ptLT_ori.y() - ptLT.y());

		int xShift = xdelta/m_Anisotropy[0];
		int yShift = ydelta/m_Anisotropy[1];

		short* pVolumeExt_slice = pVolumeExt.get() + nWidth*nHeight*i;
		short* pVolume_slice = m_pVolume.get() + m_Dims[0]*m_Dims[1]*i;
		for (auto y=0; y<nHeight; y++)
		{
			if (y<yShift || y>=yShift+m_Dims[1])
			{
				for (auto x=0; x<nWidth; x++)
				{
					pVolumeExt_slice[y*nWidth+x] = 0;
				}
			}
			else
			{
				for (auto x=0; x<nWidth; x++)
				{
					if (x<xShift || x>=xShift+m_Dims[0])
					{
						pVolumeExt_slice[y*nWidth+x] = 0;
					}
					else
					{
						memcpy(pVolumeExt_slice+y*nWidth+x, pVolume_slice+m_Dims[0]*(y-yShift), m_Dims[0]*sizeof(short));
						x += m_Dims[0];
					}
				}
			}
		}
	}

	m_Dims[0] = nWidth;
	m_Dims[1] = nHeight;
	m_dirZ = dirNorm;
	m_pVolume = pVolumeExt;
}

bool VolumeInfo::LoadVolumeFile( const char* szFile, int nWidth, int nHeight, int nDepth )
{
	StopWatch sw("VolumeInfo::LoadVolumeFile");

	if (nWidth<=0 || nHeight<=0 || nDepth<=0)
		return false;
	FILE* fp = fopen(szFile, "rb");
	if (NULL == fp)
		return false;
	m_Dims[0] = nWidth;
	m_Dims[1] = nHeight;
	m_Dims[2] = nDepth;
	m_pVolume.reset(new short[GetVolumeSize()]);
	fread(m_pVolume.get(), GetVolumeBytes(), 1, fp);
	fclose(fp);

	NormVolumeData();

	return true;
}

bool VolumeInfo::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	StopWatch sw("VolumeInfo::SetVolumeData");
	if (NULL==pData.get() || nWidth<=0 || nHeight<=0 || nDepth<=0)
		return false;

	m_Dims[0] = nWidth;
	m_Dims[1] = nHeight;
	m_Dims[2] = nDepth;
	m_pVolume = pData;

	NormVolumeData();

	return true;

}

bool VolumeInfo::GetPlaneInitSize( int& nWidth, int& nHeight, int& nNumber, const PlaneType& planeType )
{
	if (m_Dims[0]<=0 || m_Dims[1]<=0 || m_Dims[2]<=0)
		return false;
	if (m_Anisotropy[0]<=0 || m_Anisotropy[1]<=0 || m_Anisotropy[2]<=0)
		return false;

	double minAnisotropy = m_Anisotropy[0]<m_Anisotropy[1] ? m_Anisotropy[0]:m_Anisotropy[1];
	minAnisotropy = minAnisotropy<m_Anisotropy[2] ? minAnisotropy:m_Anisotropy[2];	

	switch (planeType)
	{
	case PlaneAxial:
	case PlaneAxialOblique:
		{
			nWidth = m_Dims[0]*m_Anisotropy[0]/minAnisotropy;
			nHeight = m_Dims[1]*m_Anisotropy[1]/minAnisotropy;
			nNumber = m_Dims[2]*m_Anisotropy[2]/minAnisotropy;
			return true;
		}
		break;
	case PlaneSagittal:
	case PlaneSagittalOblique:
		{
			nWidth = m_Dims[1]*m_Anisotropy[1]/minAnisotropy;
			nHeight = m_Dims[2]*m_Anisotropy[2]/minAnisotropy;
			nNumber = m_Dims[0]*m_Anisotropy[0]/minAnisotropy;
			return true;
		}
		break;
	case PlaneCoronal:
	case PlaneCoronalOblique:
		{
			nWidth = m_Dims[0]*m_Anisotropy[0]/minAnisotropy;
			nHeight = m_Dims[2]*m_Anisotropy[2]/minAnisotropy;
			nNumber = m_Dims[1]*m_Anisotropy[1]/minAnisotropy;
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