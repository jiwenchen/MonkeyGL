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
#include "Logger.h"

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

unsigned char IRender::AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	return m_dataMan.AddNewObjectMask(pData, nWidth, nHeight, nDepth);
}

unsigned char IRender::AddObjectMaskFile( const char* szFile )
{
	return m_dataMan.AddObjectMaskFile(szFile);
}

bool IRender::UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	return m_dataMan.UpdateObjectMask(pData, nWidth, nHeight, nDepth, nLabel);
}

void IRender::LoadVolumeFile( const char* szFile )
{
	m_dataMan.LoadVolumeFile(szFile);
}

void IRender::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	m_dataMan.SetDirection(dirX, dirY, dirZ);
}

void IRender::SetSpacing( double x, double y, double z )
{
	m_dataMan.SetSpacing(x, y, z);
}

void IRender::SetOrigin(Point3d pt)
{
	m_dataMan.SetOrigin(pt);
}

void IRender::Reset()
{
	m_dataMan.Reset();
}


bool IRender::SetTransferFunc(std::map<int, RGBA> ctrlPts)
{
	return m_dataMan.SetControlPoints_TF(ctrlPts);
}

bool IRender::SetTransferFunc(std::map<int, RGBA> ctrlPts, unsigned char nLabel)
{
	return m_dataMan.SetControlPoints_TF(ctrlPts, nLabel);
}

bool IRender::SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts)
{
	return m_dataMan.SetControlPoints_TF(rgbPts, alphaPts);
}

bool IRender::SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts, unsigned char nLabel)
{
	return m_dataMan.SetControlPoints_TF(rgbPts, alphaPts, nLabel);
}

void IRender::SetColorBackground(RGBA clrBG)
{
	m_dataMan.SetColorBackground(clrBG);
}

std::shared_ptr<short> IRender::GetVolumeData(int& nWidth, int& nHeight, int& nDepth)
{
	return m_dataMan.GetVolumeData(nWidth, nHeight, nDepth);
}

std::shared_ptr<unsigned char> IRender::GetMaskData()
{
	return m_dataMan.GetMaskData();
}

bool IRender::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	return m_dataMan.GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool IRender::GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	return false;
}

bool IRender::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	return m_dataMan.GetCrossHairPoint(x, y, planeType);
}

bool IRender::TransferImage2Voxel(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType)
{
	return m_dataMan.TransferImage2Voxel(x, y, z, xImage, yImage, planeType);
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

bool IRender::SetVRWWWL(float fWW, float fWL)
{
	return m_dataMan.SetVRWWWL(fWW, fWL);
}

bool IRender::SetVRWWWL(float fWW, float fWL, unsigned char nLabel)
{
	return m_dataMan.SetVRWWWL(fWW, fWL, nLabel);
}

bool IRender::SetObjectAlpha(float fAlpha)
{
	return m_dataMan.SetObjectAlpha(fAlpha);
}

bool IRender::SetObjectAlpha(float fAlpha, unsigned char nLabel)
{
	return m_dataMan.SetObjectAlpha(fAlpha, nLabel);
}

bool IRender::SetCPRLinePatient(std::vector<Point3d> cprLine)
{
	return m_dataMan.SetCPRLinePatient(cprLine);
}

bool IRender::SetCPRLineVoxel(std::vector<Point3d> cprLine)
{
	return m_dataMan.SetCPRLineVoxel(cprLine);
}

std::vector<Point3d> IRender::GetCPRLineVoxel()
{
	return m_dataMan.GetCPRLineVoxel();
}

bool IRender::RotateCPR(float angle, PlaneType planeType)
{
	return m_dataMan.RotateCPR(angle, planeType);
}