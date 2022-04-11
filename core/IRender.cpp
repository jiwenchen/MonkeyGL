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

#include "IRender.h"

using namespace MonkeyGL;

IRender::IRender(void)
{
}


IRender::~IRender(void)
{
}

bool IRender::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	return m_dataMan.SetVolumeData(pData, nWidth, nHeight, nDepth);
}

void IRender::SetVolumeFile( const char* szFile, int nWidth, int nHeight, int nDepth )
{
	m_dataMan.LoadVolumeFile(szFile, nWidth, nHeight, nDepth);
}

void IRender::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	m_dataMan.SetDirection(dirX, dirY, dirZ);
}

void IRender::SetAnisotropy( double x, double y, double z )
{
	m_dataMan.SetAnisotropy(x, y, z);
}

void IRender::Reset()
{
	m_dataMan.Reset();
}

void IRender::SetTransferFunc( const std::map<int, RGBA>& ctrlPoints )
{
	m_dataMan.SetControlPoints_TF(ctrlPoints);
}

void IRender::SetTransferFunc( const std::map<int, RGBA>& rgbPoints, const std::map<int, double>& alphaPoints)
{
	m_dataMan.SetControlPoints_TF(rgbPoints, alphaPoints);
}

void IRender::SetColorBackground(float clrBkg[])
{
	m_dataMan.SetColorBackground(clrBkg);
}

std::shared_ptr<short> IRender::GetVolumeData(int& nWidth, int& nHeight, int& nDepth)
{
	return m_dataMan.GetVolumeData(nWidth, nHeight, nDepth);
}

bool IRender::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	return m_dataMan.GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool IRender::GetPlaneData(short* pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	return false;
}

bool IRender::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	return m_dataMan.GetCrossHairPoint(x, y, planeType);
}

bool IRender::TransferImage2Object(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType)
{
	return m_dataMan.TransferImage2Object(x, y, z, xImage, yImage, planeType);
}

bool IRender::GetCrossHairPoint3D( Point3d& pt )
{
	pt = m_dataMan.GetCrossHair();
	return true;
}

bool IRender::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
{
	return m_dataMan.GetDirection(dirH, dirV, planeType);
}

bool IRender::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	return m_dataMan.GetDirection3D(dir3dH, dir3dV, planeType);
}

bool IRender::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType )
{
	return m_dataMan.GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void IRender::Browse( float fDelta, PlaneType planeType )
{
	m_dataMan.Browse(fDelta, planeType);
}

void IRender::SetPlaneIndex( int index, PlaneType planeType )
{
	m_dataMan.SetPlaneIndex(index, planeType);
}

void IRender::PanCrossHair( int nx, int ny, PlaneType planeType )
{
	m_dataMan.PanCrossHair(nx, ny, planeType);
}

void IRender::RotateCrossHair( float fAngle, PlaneType planeType )
{
	m_dataMan.RotateCrossHair(fAngle, planeType);
}

double IRender::GetPixelSpacing( PlaneType planeType )
{
	return m_dataMan.GetPixelSpacing(planeType);
}

bool IRender::GetPlaneIndex( int& index, PlaneType planeType )
{
	return m_dataMan.GetPlaneIndex(index, planeType);
}

bool IRender::GetPlaneNumber( int& nTotalNum, PlaneType planeType )
{
	return m_dataMan.GetPlaneNumber(nTotalNum, planeType);
}

bool IRender::GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType )
{
	return m_dataMan.GetPlaneRotateMatrix(pMatirx, planeType);
}

void IRender::UpdateThickness( double val )
{
	m_dataMan.UpdateThickness(val);
}

void IRender::SetThickness(double val, PlaneType planeType)
{
	m_dataMan.SetThickness(val, planeType);
}

bool IRender::GetThickness(double& val, PlaneType planeType)
{
	return m_dataMan.GetThickness(val, planeType);
}

void IRender::SetMPRType(MPRType type)
{
	m_dataMan.SetMPRType(type);
}
