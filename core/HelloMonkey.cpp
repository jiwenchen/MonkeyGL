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

#include "HelloMonkey.h"
#include "Render.h"
#include <memory>
#include "Base64.hpp"
#include "StopWatch.h"
#include "fpng/fpng.h"
#include "Logger.h"
#include "Methods.h"

using namespace MonkeyGL;

HelloMonkey::HelloMonkey()
{
	Logger::Init();

	Logger::Info("MonkeyGL has started....");

	fpng::fpng_init();
	if (fpng::fpng_cpu_supports_sse41()){
		Logger::Info("fpng cpu supports sse41");
	}
	else {
		Logger::Error("fpng cpu not supports sse41");
	}

	m_nWidth_VR = 512;
	m_nHeight_VR = 512;

	_pRender.reset(new Render());
	m_bShowCPRLineInVR = false;
}

HelloMonkey::~HelloMonkey(void)
{
}

void HelloMonkey::SetLogLevel(LogLevel level)
{
	Logger::SetLevel(level);
}

bool HelloMonkey::SetTransferFunc( std::map<int, RGBA> ctrlPoints )
{	
	if (!_pRender)
		return false;
	return _pRender->SetTransferFunc(ctrlPoints);
}

bool HelloMonkey::SetTransferFunc( std::map<int, RGBA> ctrlPoints, unsigned char nLabel )
{
	if (!_pRender)
		return false;
	
	return _pRender->SetTransferFunc(ctrlPoints, nLabel);
}

bool HelloMonkey::SetTransferFunc( std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints)
{
	if (!_pRender)
		return false;

	return _pRender->SetTransferFunc(rgbPoints, alphaPoints);
}

bool HelloMonkey::SetTransferFunc( std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints, unsigned char nLabel)
{
	if (!_pRender)
		return false;

	return _pRender->SetTransferFunc(rgbPoints, alphaPoints, nLabel);
}

bool HelloMonkey::LoadTransferFunction(const char* szFile)
{
	if (!_pRender)
		return false;

	return _pRender->LoadTransferFunction(szFile);
}

bool HelloMonkey::SaveTransferFunction(const char* szFile)
{
	if (!_pRender)
		return false;

	return _pRender->SaveTransferFunction(szFile);
}

void HelloMonkey::SetColorBackground(RGBA clrBG)
{
	if (!_pRender)
		return;

	_pRender->SetColorBackground(clrBG);
}

void HelloMonkey::LoadVolumeFile( const char* szFile )
{
	if (!_pRender)
		return;

	_pRender->LoadVolumeFile(szFile);
}

void HelloMonkey::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	if (!_pRender)
		return;
	_pRender->SetDirection(dirX, dirY, dirZ);
}

void HelloMonkey::SetSpacing( double x, double y, double z )
{
	if (!_pRender)
		return;
	_pRender->SetSpacing(x, y, z);
}

void HelloMonkey::SetOrigin(Point3d pt)
{
	if (!_pRender)
		return;
	_pRender->SetOrigin(pt);
}

void HelloMonkey::Reset()
{
	if (!_pRender)
		return;
	_pRender->Reset();
}

bool HelloMonkey::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	if (!_pRender)
		return false;
	return _pRender->SetVolumeData(pData, nWidth, nHeight, nDepth);
}

unsigned char HelloMonkey::AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	if (!_pRender)
		return 0;
	return _pRender->AddNewObjectMask(pData, nWidth, nHeight, nDepth);
}

unsigned char HelloMonkey::AddObjectMaskFile(const char* szFile)
{
	if (!_pRender)
		return 0;
	return _pRender->AddObjectMaskFile(szFile);
}

bool HelloMonkey::UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	if (!_pRender)
		return false;
	return _pRender->UpdateObjectMask(pData, nWidth, nHeight, nDepth, nLabel);
}

std::shared_ptr<short> HelloMonkey::GetVolumeData(int& nWidth, int& nHeight, int& nDepth)
{
	if (!_pRender)
		return NULL;
	return _pRender->GetVolumeData(nWidth, nHeight, nDepth);
}

bool HelloMonkey::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	if (!_pRender)
		return false;
	return _pRender->GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool HelloMonkey::GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	if (!_pRender)
		return NULL;
	return _pRender->GetPlaneData(pData, nWidth, nHeight, planeType);
}

std::string HelloMonkey::GetPlaneData_pngString(const PlaneType& planeType)
{
	if (!_pRender)
		return "";

	StopWatch sw("GetPlaneData_pngString[%s]", PlaneTypeName(planeType).c_str());

	int nWidth = 0, nHeight = 0;
 	std::shared_ptr<short> pData;
	{
		StopWatch sw("GetPlaneData");
		if (!_pRender->GetPlaneData(pData, nWidth, nHeight, planeType))
			return "";
	}

	std::vector<uint8_t> out_buf;
	{
		StopWatch sw("fpng");
		fpng::fpng_encode_image_to_memory(
			(void*)pData.get(),
			nWidth/2,
			nHeight,
			4,
			out_buf
		);
		Logger::Info(
			"plane encode, from %d to %d, ratio %.4f",
			nWidth*nHeight*sizeof(short),
			out_buf.size(),
			1.0*out_buf.size()/(nWidth*nHeight*sizeof(short))
		);
	}

	std::string strBase64 = "";
	{
		StopWatch sw("Base64 Encode");
		strBase64 = Base64::Encode(out_buf.data(), out_buf.size());

		Logger::Info(
			"plane base64, from %d to %d, ratio %.4f",
			out_buf.size(),
			strBase64.length(),
			1.0*strBase64.length()/out_buf.size()
		);
	}

	return strBase64;
}

std::string HelloMonkey::GetOriginData_pngString(int slice)
{
	if (!_pRender)
		return "";

	StopWatch sw("GetOriginData_pngString");

	int nWidth = 0, nHeight = 0, nDepth = 0;
	std::shared_ptr<short> pData = GetVolumeData(nWidth, nHeight, nDepth);
	std::shared_ptr<unsigned char> pMask = _pRender->GetMaskData();

	if (slice < 0)
		slice = 0;
	else if (slice >= nDepth)
		slice = nDepth-1;

	std::shared_ptr<short> pSliceData(new short[nWidth*nHeight]);
	memcpy(pSliceData.get(), pData.get()+nWidth*nHeight*slice, nWidth*nHeight*sizeof(short));
	unsigned char* pSliceMask = pMask.get()+nWidth*nHeight*slice;
	if (pMask){
		for (int i=0; i<nWidth*nHeight; i++){
			if (pSliceMask[i] == 0){
				pSliceData.get()[i] = -2048;
			}
		}
	}

	std::vector<uint8_t> out_buf;
	{
		StopWatch sw("fpng");
		fpng::fpng_encode_image_to_memory(
			(void*)(pSliceData.get()),
			nWidth/2,
			nHeight,
			4,
			out_buf
		);
		Logger::Info(
			"plane encode, from %d to %d, ratio %.4f",
			nWidth*nHeight*sizeof(short),
			out_buf.size(),
			1.0*out_buf.size()/(nWidth*nHeight*sizeof(short))
		);
	}

	std::string strBase64 = "";
	{
		StopWatch sw("Base64 Encode");
		strBase64 = Base64::Encode(out_buf.data(), out_buf.size());

		Logger::Info(
			"plane base64, from %d to %d, ratio %.4f",
			out_buf.size(),
			strBase64.length(),
			1.0*strBase64.length()/out_buf.size()
		);
	}

	return strBase64;
}

bool HelloMonkey::GetVRData( unsigned char* pVR, int nWidth, int nHeight )
{
	if (!_pRender)
		return false;
	if (!_pRender->GetVRData(pVR, nWidth, nHeight))
		return false;

	return true;	
}


std::vector<uint8_t> HelloMonkey::GetVRData_png(int nWidth, int nHeight)
{
	StopWatch sw("GetVRData_png");
	std::vector<uint8_t> out_buf;
	if (!_pRender)
		return out_buf;
	
 	std::shared_ptr<unsigned char> pVR (new unsigned char[nWidth*nHeight*3]);

	{
		StopWatch sw("GetVRData");
		if (!_pRender->GetVRData(pVR.get(), nWidth, nHeight))
			return out_buf;

		if (m_bShowCPRLineInVR)
		{
			std::vector<Point3d> cprLine = _pRender->GetCPRLineVoxel();
			StopWatch sw("cpr line size: %d", cprLine.size());
			if (cprLine.size() >= 2){
				float x0, y0, x1, y1;
				_pRender->TransferVoxel2ImageInVR(x0, y0, nWidth, nHeight, cprLine[0]);
				for (int i=1; i<cprLine.size(); i++){
					_pRender->TransferVoxel2ImageInVR(x1, y1, nWidth, nHeight, cprLine[i]);
					Methods::DrawLineInImage24Bit(pVR.get(), nWidth, nHeight, x0, y0, x1, y1, 1);
					x0 = x1;
					y0 = y1;
				}
			}
		}
	}
	{
		StopWatch sw("fpng");
		fpng::fpng_encode_image_to_memory(
			(void*)pVR.get(),
			nWidth,
			nHeight,
			3,
			out_buf
		);

		Logger::Info(
			"vr encode, from %d to %d, ratio %.4f",
			nWidth*nHeight*3,
			out_buf.size(),
			1.0*out_buf.size()/(nWidth*nHeight*3)
		);
	}

	return out_buf;
}

void HelloMonkey::SaveVR2Png(const char* szFile, int nWidth, int nHeight)
{
	std::vector<uint8_t> out_buf = GetVRData_png(nWidth, nHeight);

	FILE* fp = fopen(szFile, "wb");
	if (NULL == fp)
	{
		Logger::Error("failed to save png file [%s]", szFile);
	}
	fwrite(out_buf.data(), 1, nWidth*nHeight*3, fp);
	fclose(fp);
	Logger::Info("saved png file [%s]", szFile);
}

bool HelloMonkey::SetVRSize(int nWidth, int nHeight)
{
	float delta = (nWidth*nHeight)/(768*768);
	if (delta > 1){
		m_nWidth_VR = nWidth / delta;
		m_nHeight_VR = nHeight / delta;
	}
	else{
		m_nWidth_VR = nWidth;
		m_nHeight_VR = nHeight;
	}

	if (m_nWidth_VR % 2){
		m_nWidth_VR = m_nWidth_VR + 1;
	}
	
	return true;
}

std::string HelloMonkey::GetVRData_pngString()
{
	StopWatch sw("GetVRData_pngString");
	std::vector<uint8_t> out_buf = GetVRData_png(m_nWidth_VR, m_nHeight_VR);

	std::string strBase64 = "";
	{
		StopWatch sw("Base64 Encode");
		strBase64 = Base64::Encode(out_buf.data(), out_buf.size());

		Logger::Info(
			"vr base64, from %d to %d, ratio %.4f",
			out_buf.size(),
			strBase64.length(),
			1.0*strBase64.length()/out_buf.size()
		);
	}

	return strBase64;
}

bool HelloMonkey::GetBatchData( std::vector<short*>& vecBatchData, const BatchInfo& batchInfo )
{
	if (!_pRender)
		return NULL;
	return _pRender->GetBatchData(vecBatchData, batchInfo);
}

bool HelloMonkey::GetPlaneIndex( int& index, PlaneType planeType )
{
	if (!_pRender)
		return NULL;
	return _pRender->GetPlaneIndex(index, planeType);
}

bool HelloMonkey::GetPlaneNumber( int& nTotalNum, PlaneType planeType )
{
	if (!_pRender)
		return NULL;
	return _pRender->GetPlaneNumber(nTotalNum, planeType);
}

bool HelloMonkey::GetPlaneRotateMatrix( float* pMatrix, PlaneType planeType )
{
	if (!_pRender)
		return NULL;
	return _pRender->GetPlaneRotateMatrix(pMatrix, planeType);
}

void HelloMonkey::Anterior()
{
	if (!_pRender)
		return;
	_pRender->Anterior();
}

void HelloMonkey::Posterior()
{
	if (!_pRender)
		return;
	_pRender->Posterior();
}

void HelloMonkey::Left()
{
	if (!_pRender)
		return;
	_pRender->Left();
}

void HelloMonkey::Right()
{
	if (!_pRender)
		return;
	_pRender->Right();
}

void HelloMonkey::Head()
{
	if (!_pRender)
		return;
	_pRender->Head();
}

void HelloMonkey::Foot()
{
	if (!_pRender)
		return;
	_pRender->Foot();
}

void HelloMonkey::SetRenderType(RenderType type)
{
	if (!_pRender)
		return;
	_pRender->SetRenderType(type);
}

void HelloMonkey::Rotate( float fxRotate, float fyRotate )
{
	if (!_pRender)
		return;
	_pRender->Rotate(fxRotate, fyRotate);
}

float HelloMonkey::Zoom( float ratio )
{
	if (!_pRender)
		return 0.0f;
	return _pRender->Zoom(ratio);
}

float HelloMonkey::GetZoomRatio()
{
	if (!_pRender)
		return 0.0f;
	return _pRender->GetZoomRatio();
}

void HelloMonkey::Pan( float fxShift, float fyShift )
{
	if (!_pRender)
		return;
	_pRender->Pan(fxShift, fyShift);
}

bool HelloMonkey::SetVRWWWL(float fWW, float fWL)
{
	if (!_pRender)
		return false;
	return _pRender->SetVRWWWL(fWW, fWL);
}

bool HelloMonkey::SetVRWWWL(float fWW, float fWL, unsigned char nLabel)
{
	if (!_pRender)
		return false;
	return _pRender->SetVRWWWL(fWW, fWL, nLabel);
}

bool HelloMonkey::SetObjectAlpha(float fAlpha)
{
	if (!_pRender)
		return false;
	return _pRender->SetObjectAlpha(fAlpha);
}

bool HelloMonkey::SetObjectAlpha(float fAlpha, unsigned char nLabel)
{
	if (!_pRender)
		return false;
	return _pRender->SetObjectAlpha(fAlpha, nLabel);
}

void HelloMonkey::Browse( float fDelta, PlaneType planeType )
{
	if (!_pRender)
		return;
	_pRender->Browse(fDelta, planeType);
}

void HelloMonkey::PanCrossHair( float fx, float fy, PlaneType planeType )
{
	if (!_pRender)
		return;
	_pRender->PanCrossHair(fx, fy, planeType);
}

bool HelloMonkey::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	if (!_pRender)
		return false;
	return _pRender->GetCrossHairPoint(x, y, planeType);
}

Point2d HelloMonkey::GetCrossHairPoint(const PlaneType& planeType)
{
	Point2d pt(-1, -1);

	if (!_pRender)
		return pt;
	double x = -1, y = -1;
	if (_pRender->GetCrossHairPoint(x, y, planeType)){
		pt = Point2d(x, y);
	}
	return pt;
}

bool HelloMonkey::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
{
	if (!_pRender)
		return false;
	return _pRender->GetDirection(dirH, dirV, planeType);
}

bool HelloMonkey::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	if (!_pRender)
		return false;
	return _pRender->GetDirection3D(dir3dH, dir3dV, planeType);
}

bool HelloMonkey::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType )
{
	if (!_pRender)
		return false;
	return _pRender->GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void HelloMonkey::RotateCrossHair( float fAngle, PlaneType planeType )
{
	if (!_pRender)
		return;
	_pRender->RotateCrossHair(fAngle, planeType);
}

void HelloMonkey::SetPlaneIndex( int index, PlaneType planeType )
{
	if (!_pRender)
		return;
	_pRender->SetPlaneIndex(index, planeType);
}

double HelloMonkey::GetPixelSpacing( PlaneType planeType )
{
	if (!_pRender)
		return 1.0;
	return _pRender->GetPixelSpacing(planeType);
}

bool HelloMonkey::TransferImage2Voxel( double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType )
{
	if (!_pRender)
		return false;
	return _pRender->TransferImage2Voxel( x, y, z, xImage, yImage, planeType );
}

bool HelloMonkey::GetCrossHairPoint3D( Point3d& pt )
{
	if (!_pRender)
		return false;
	return _pRender->GetCrossHairPoint3D( pt );
}

void HelloMonkey::UpdateThickness( double val )
{
	if (!_pRender)
		return;
	return _pRender->UpdateThickness( val );
}

void HelloMonkey::SetThickness(double val, PlaneType planeType)
{
	if (!_pRender)
		return;
	return _pRender->SetThickness(val, planeType);
}

double HelloMonkey::GetThickness(PlaneType planeType)
{
	if (!_pRender)
		return false;
	return _pRender->GetThickness(planeType);
}

void HelloMonkey::SetMPRType( MPRType type )
{
	if (!_pRender)
		return;
	return _pRender->SetMPRType(type);
}

bool HelloMonkey::SetCPRLinePatient(std::vector<Point3d> cprLine)
{
	if (!_pRender)
		return false;
	return _pRender->SetCPRLinePatient(cprLine);
}

bool HelloMonkey::SetCPRLineVoxel(std::vector<Point3d> cprLine)
{
	if (!_pRender)
		return false;
	return _pRender->SetCPRLineVoxel(cprLine);
}

bool HelloMonkey::RotateCPR(float angle, PlaneType planeType)
{
	if (!_pRender)
		return false;
	return _pRender->RotateCPR(angle, planeType);
}

void HelloMonkey::ShowCPRLineInVR(bool bShow)
{
	m_bShowCPRLineInVR = bShow;
}

void HelloMonkey::ShowPlaneInVR(bool bShow)
{
	if (!_pRender)
		return;
	return _pRender->ShowPlaneInVR(bShow);
}
