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
#include "DataManager.h"

using namespace MonkeyGL;

IRender::IRender(void)
{
}

IRender::~IRender(void)
{
}

bool IRender::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	return DataManager::Instance()->SetVolumeData(pData, nWidth, nHeight, nDepth);
}

unsigned char IRender::AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	return DataManager::Instance()->AddNewObjectMask(pData, nWidth, nHeight, nDepth);
}

unsigned char IRender::AddObjectMaskFile( const char* szFile )
{
	return DataManager::Instance()->AddObjectMaskFile(szFile);
}

bool IRender::UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	return DataManager::Instance()->UpdateObjectMask(pData, nWidth, nHeight, nDepth, nLabel);
}

void IRender::LoadVolumeFile( const char* szFile )
{
	DataManager::Instance()->LoadVolumeFile(szFile);
}

void IRender::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	DataManager::Instance()->SetDirection(dirX, dirY, dirZ);
}

void IRender::SetSpacing( double x, double y, double z )
{
	DataManager::Instance()->SetSpacing(x, y, z);
}

void IRender::SetOrigin(Point3d pt)
{
	DataManager::Instance()->SetOrigin(pt);
}

void IRender::Reset()
{
	DataManager::Instance()->Reset();
}

bool IRender::SetTransferFunc(std::map<int, RGBA> ctrlPts)
{
	return DataManager::Instance()->SetControlPoints_TF(ctrlPts);
}

bool IRender::SetTransferFunc(std::map<int, RGBA> ctrlPts, unsigned char nLabel)
{
	return DataManager::Instance()->SetControlPoints_TF(ctrlPts, nLabel);
}

bool IRender::SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts)
{
	return DataManager::Instance()->SetControlPoints_TF(rgbPts, alphaPts);
}

bool IRender::SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts, unsigned char nLabel)
{
	return DataManager::Instance()->SetControlPoints_TF(rgbPts, alphaPts, nLabel);
}

bool IRender::LoadTransferFunction(const char* szFile)
{
	return DataManager::Instance()->LoadTransferFunction(szFile);
}

bool IRender::SaveTransferFunction(const char* szFile)
{
	return DataManager::Instance()->SaveTransferFunction(szFile);
}

void IRender::SetColorBackground(RGBA clrBG)
{
	DataManager::Instance()->SetColorBackground(clrBG);
}

std::shared_ptr<short> IRender::GetVolumeData(int& nWidth, int& nHeight, int& nDepth)
{
	return DataManager::Instance()->GetVolumeData(nWidth, nHeight, nDepth);
}

std::shared_ptr<unsigned char> IRender::GetMaskData()
{
	return DataManager::Instance()->GetMaskData();
}

bool IRender::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	return DataManager::Instance()->GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool IRender::GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	return false;
}

bool IRender::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	return DataManager::Instance()->GetCrossHairPoint(x, y, planeType);
}

bool IRender::TransferImage2Voxel(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType)
{
	return DataManager::Instance()->TransferImage2Voxel(x, y, z, xImage, yImage, planeType);
}

bool IRender::GetCrossHairPoint3D( Point3d& pt )
{
	pt = DataManager::Instance()->GetCrossHair();
	return true;
}

bool IRender::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
{
	return DataManager::Instance()->GetDirection(dirH, dirV, planeType);
}

bool IRender::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	return DataManager::Instance()->GetDirection3D(dir3dH, dir3dV, planeType);
}

bool IRender::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType )
{
	return DataManager::Instance()->GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void IRender::Browse( float fDelta, PlaneType planeType )
{
	DataManager::Instance()->Browse(fDelta, planeType);
}

void IRender::SetPlaneIndex( int index, PlaneType planeType )
{
	DataManager::Instance()->SetPlaneIndex(index, planeType);
}

void IRender::PanCrossHair( float fx, float fy, PlaneType planeType )
{
	DataManager::Instance()->PanCrossHair(fx, fy, planeType);
}

void IRender::RotateCrossHair( float fAngle, PlaneType planeType )
{
	DataManager::Instance()->RotateCrossHair(fAngle, planeType);
}

double IRender::GetPixelSpacing( PlaneType planeType )
{
	return DataManager::Instance()->GetPixelSpacing(planeType);
}

bool IRender::SetVRSize(int nWidth, int nHeight)
{
	return DataManager::Instance()->SetVRSize(nWidth, nHeight);
}

void IRender::GetVRSize(int& nWidth, int& nHeight)
{
	DataManager::Instance()->GetVRSize(nWidth, nHeight);
}

bool IRender::GetPlaneIndex( int& index, PlaneType planeType )
{
	return DataManager::Instance()->GetPlaneIndex(index, planeType);
}

bool IRender::GetPlaneNumber( int& nTotalNum, PlaneType planeType )
{
	return DataManager::Instance()->GetPlaneNumber(nTotalNum, planeType);
}

bool IRender::GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType )
{
	return DataManager::Instance()->GetPlaneRotateMatrix(pMatirx, planeType);
}

void IRender::Anterior()
{
	DataManager::Instance()->Anterior();
}

void IRender::Posterior()
{
	DataManager::Instance()->Posterior();
}

void IRender::Left()
{
	DataManager::Instance()->Left();
}

void IRender::Right()
{
	DataManager::Instance()->Right();
}

void IRender::Head()
{
	DataManager::Instance()->Head();
}

void IRender::Foot()
{
	DataManager::Instance()->Foot();
}

void IRender::Rotate(float fxRotate, float fyRotate)
{
	DataManager::Instance()->Rotate(fxRotate, fyRotate);
}

float IRender::Zoom(float ratio)
{
	return DataManager::Instance()->Zoom(ratio);
}

float IRender::GetZoomRatio()
{
	return DataManager::Instance()->GetZoomRatio();
}

void IRender::Pan(float fxShift, float fyShift)
{
	DataManager::Instance()->Pan(fxShift, fyShift);
}

void IRender::UpdateThickness( double val )
{
	DataManager::Instance()->UpdateThickness(val);
}

void IRender::SetThickness(double val, PlaneType planeType)
{
	DataManager::Instance()->SetThickness(val, planeType);
}

double IRender::GetThickness(PlaneType planeType)
{
	return DataManager::Instance()->GetThickness(planeType);
}

void IRender::SetMPRType(MPRType type)
{
	DataManager::Instance()->SetMPRType(type);
}

bool IRender::TransferVoxel2ImageInVR(float& fx, float& fy, int nWidth, int nHeight, Point3d ptVoxel)
{
	return DataManager::Instance()->TransferVoxel2ImageInVR(fx, fy, nWidth, nHeight, ptVoxel);
}

void IRender::SetRenderType(RenderType type)
{
	DataManager::Instance()->SetRenderType(type);
}

bool IRender::SetVRWWWL(float fWW, float fWL)
{
	return DataManager::Instance()->SetVRWWWL(fWW, fWL);
}

bool IRender::SetVRWWWL(float fWW, float fWL, unsigned char nLabel)
{
	return DataManager::Instance()->SetVRWWWL(fWW, fWL, nLabel);
}

bool IRender::SetObjectAlpha(float fAlpha)
{
	return DataManager::Instance()->SetObjectAlpha(fAlpha);
}

bool IRender::SetObjectAlpha(float fAlpha, unsigned char nLabel)
{
	return DataManager::Instance()->SetObjectAlpha(fAlpha, nLabel);
}

bool IRender::SetCPRLinePatient(std::vector<Point3d> cprLine)
{
	return DataManager::Instance()->SetCPRLinePatient(cprLine);
}

bool IRender::SetCPRLineVoxel(std::vector<Point3d> cprLine)
{
	return DataManager::Instance()->SetCPRLineVoxel(cprLine);
}

std::vector<Point3d> IRender::GetCPRLineVoxel()
{
	return DataManager::Instance()->GetCPRLineVoxel();
}

bool IRender::RotateCPR(float angle, PlaneType planeType)
{
	return DataManager::Instance()->RotateCPR(angle, planeType);
}

void IRender::ShowPlaneInVR(bool bShow)
{
	return DataManager::Instance()->ShowPlaneInVR(bShow);
}

bool IRender::AddAnnotation(PlaneType planeType, std::string txt, int x, int y, FontSize fontSize, AnnotationFormat annoFormat, RGB clr)
{
	return DataManager::Instance()->AddAnnotation(planeType, txt, x, y, fontSize, annoFormat, clr);
}

bool IRender::RemovePlaneAnnotations(PlaneType planeType)
{
	return DataManager::Instance()->RemovePlaneAnnotations(planeType);
}

bool IRender::RemoveAllAnnotations()
{
	return DataManager::Instance()->RemoveAllAnnotations();
}

bool IRender::EnableLayer(LayerType layerType, bool bEnable)
{
	return DataManager::Instance()->EnableLayer(layerType, bEnable);
}

void IRender::SetCPRLineColor(RGB clr)
{
	DataManager::Instance()->SetCPRLineColor(clr);
}