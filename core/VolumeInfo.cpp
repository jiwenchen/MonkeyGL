// MIT License

// Copyright (c) 2022-2023 jiwenchen(cjwbeyond@hotmail.com)

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
#include <cstring>
#include "StopWatch.h"
#include "Logger.h"
#include "Methods.h"
#include "ImageReader.h"

using namespace MonkeyGL;

VolumeInfo::VolumeInfo( void )
{
	m_pVolume.reset();
	m_pMask.reset();
	m_bVolumeHasInverted = false;
	memset(m_Dims, 0, 3*sizeof(int));
	m_Spacing[0] = 1.0;
	m_Spacing[1] = 1.0;
	m_Spacing[2] = 1.0;
	m_dirX = Direction3d(1, 0, 0);
	m_dirY = Direction3d(0, 1, 0);
	m_dirZ = Direction3d(0, 0, -1); // raw data from head to foot
}

VolumeInfo::~VolumeInfo( void )
{
}

void VolumeInfo::Clear(){
	m_pVolume.reset();
	m_pMask.reset();
	m_bVolumeHasInverted = false;
}

bool VolumeInfo::LoadVolumeFile( const char* szFile)
{
	StopWatch sw("VolumeInfo::LoadVolumeFile");
	if (!ImageReader::Read(
			szFile,
			m_pVolume,
			m_Dims,
			m_Spacing,
			m_ptOriginPatient,
			m_dirX,
			m_dirY,
			m_dirZ
		)
	){
		return false;
	}

	m_pMask.reset();

	return true;
}

bool VolumeInfo::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	StopWatch sw("VolumeInfo::SetVolumeData");
	if (!pData || nWidth<=0 || nHeight<=0 || nDepth<=0)
		return false;

	m_Dims[0] = nWidth;
	m_Dims[1] = nHeight;
	m_Dims[2] = nDepth;
	m_pVolume = pData;
	m_pMask.reset();

	return true;
}

bool VolumeInfo::AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	if (!pData){
		return false;
	}
	int nTotalVoxel = nWidth * nHeight * nDepth;
	if (!m_pMask){
		m_pMask.reset(new unsigned char[nTotalVoxel]);
		for (int i=0; i<nTotalVoxel; i++){
			if (pData.get()[i] > 0){
				m_pMask.get()[i] = nLabel;
			}
		}
	}
	else{
		for (int i=0; i<nWidth*nHeight*nDepth; i++){
			if (pData.get()[i] > 0){
				m_pMask.get()[i] = nLabel;
			}
		}
	}
	return true;
}

bool VolumeInfo::UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	if (!pData || !m_pMask){
		return false;
	}

	for (int i=0; i<nWidth*nHeight*nDepth; i++){
		if (pData.get()[i] > 0){
			m_pMask.get()[i] = nLabel;
		}
		else if (pData.get()[i] == 0 && m_pMask.get()[i] == nLabel){
			m_pMask.get()[i] = 0;
		}
	}
	return true;
}

void VolumeInfo::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	m_dirX = dirX;
	m_dirY = dirY;
	m_dirZ = dirZ;
}

bool VolumeInfo::Need2InvertZ()
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
	if (!m_pVolume)
		return;

	if (Need2InvertZ())
	{
		int nSizeSlice = m_Dims[0] * m_Dims[1];
		std::shared_ptr<short> pslice(new short[nSizeSlice]);
		for (int i=0; i<m_Dims[2]/2; i++)
		{
			memcpy(pslice.get(), m_pVolume.get() + nSizeSlice * i, nSizeSlice * sizeof(short));
			memcpy(m_pVolume.get() + nSizeSlice * i, m_pVolume.get() + nSizeSlice * (m_Dims[2]-1-i), nSizeSlice * sizeof(short));
			memcpy(m_pVolume.get() + nSizeSlice * (m_Dims[2] - 1 - i), pslice.get(), nSizeSlice * sizeof(short));
		}

		m_dirZ = Direction3d(-m_dirZ.x(), -m_dirZ.y(), -m_dirZ.z());
		m_bVolumeHasInverted = true;
	}

	if (IsPerpendicularCoord())
		return;

	Direction3d dirNorm = m_dirX.cross(m_dirY);
	double xLen = m_Spacing[0] * m_Dims[0];
	double yLen = m_Spacing[1] * m_Dims[1];
	double zLen = m_Spacing[2] * m_Dims[2];

	double cosV = abs(dirNorm.dot(m_dirZ));
	if (cosV == 0)
	{
		return;
	}

	Point3d ptLeftTop_StartSlice(0, 0, 0);
	Point3d ptLeftTop_EndSlice = ptLeftTop_StartSlice + Point3d(m_dirZ.x()*zLen/cosV, m_dirZ.y()*zLen/cosV, m_dirZ.z()*zLen/cosV);
	Point3d ptProj_LeftTop_EndSlice = Methods::Projection_Point2Plane(ptLeftTop_EndSlice, dirNorm, ptLeftTop_StartSlice);

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

	int nWidth = xLen / m_Spacing[0];
	int nHeight = yLen / m_Spacing[1];

	//m_Dims[0] = nWidth;
	//m_Dims[1] = nHeight;
	std::shared_ptr<short> pVolumeExt(new short[nWidth*nHeight*m_Dims[2]]);
	for (auto i=0; i<m_Dims[2]; i++)
	{
		double zdelta = i*m_Spacing[2];
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

		int xShift = xdelta/m_Spacing[0];
		int yShift = ydelta/m_Spacing[1];

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

std::shared_ptr<unsigned char> VolumeInfo::NormMaskData(std::shared_ptr<unsigned char>pData)
{
	if (!pData)
		return pData;

	if (m_bVolumeHasInverted)
	{
		int nSizeSlice = m_Dims[0] * m_Dims[1];
		std::shared_ptr<unsigned char> pslice(new unsigned char[nSizeSlice]);
		for (int i=0; i<m_Dims[2]/2; i++)
		{
			memcpy(pslice.get(), pData.get() + nSizeSlice * i, nSizeSlice * sizeof(unsigned char));
			memcpy(pData.get() + nSizeSlice * i, pData.get() + nSizeSlice * (m_Dims[2]-1-i), nSizeSlice * sizeof(unsigned char));
			memcpy(pData.get() + nSizeSlice * (m_Dims[2] - 1 - i), pslice.get(), nSizeSlice * sizeof(unsigned char));
		}
	}

	return pData;
}

std::shared_ptr<unsigned char> VolumeInfo::CheckAndNormMaskData(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	if (!pData || nWidth<=0 || nHeight<=0 || nDepth<=0)
		return std::shared_ptr<unsigned char>(NULL);

	if (nWidth != m_Dims[0] || nHeight != m_Dims[1] || nDepth != m_Dims[2])
	{
		Logger::Warn("invalid mask size[%d, %d, %d], to volume size[%d, %d, %d]", m_Dims[0], m_Dims[1], m_Dims[2], nWidth, nHeight, nDepth);
		return std::shared_ptr<unsigned char>(NULL);
	}
	pData = NormMaskData(pData);
	return pData;
}

void VolumeInfo::SetSpacing(double x, double y, double z)
{
	m_Spacing[0] = x;
	m_Spacing[1] = y;
	m_Spacing[2] = z;
}

Point3d VolumeInfo::Patient2Voxel(Point3d ptPatient)
{	
	double x = Methods::Length_VectorInLine(ptPatient, m_dirX, m_ptOriginPatient) / m_Spacing[0];
	double y = Methods::Length_VectorInLine(ptPatient, m_dirY, m_ptOriginPatient) / m_Spacing[1];
	double z = Methods::Length_VectorInLine(ptPatient, m_dirZ, m_ptOriginPatient) / m_Spacing[2];
	return Point3d(x, y, z);
}

Point3d VolumeInfo::Voxel2Patient(Point3d ptVoxel)
{
	double x = ptVoxel.x() * m_Spacing[0];
	double y = ptVoxel.y() * m_Spacing[1];
	double z = ptVoxel.z() * m_Spacing[2];
	return m_ptOriginPatient + m_dirX * x + m_dirY * y + m_dirZ * z;
}