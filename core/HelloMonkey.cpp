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
#include "AnnotationUtils.h"

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

	m_pRender.reset(new Render());
	m_bShowCPRLineInVR = false;

	AnnotationUtils::Init();
}

void HelloMonkey::Transfer2Base64(unsigned char* pData, int nWidth, int nHeight)
{	
	std::string strBase64 = "";
	{
		strBase64 = Base64::Encode(pData, nWidth*nHeight);
	}
	printf("\"%s\",\n", strBase64.c_str());
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
	if (!m_pRender)
		return false;
	return m_pRender->SetTransferFunc(ctrlPoints);
}

bool HelloMonkey::SetTransferFunc( std::map<int, RGBA> ctrlPoints, unsigned char nLabel )
{
	if (!m_pRender)
		return false;
	
	return m_pRender->SetTransferFunc(ctrlPoints, nLabel);
}

bool HelloMonkey::SetTransferFunc( std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints)
{
	if (!m_pRender)
		return false;

	return m_pRender->SetTransferFunc(rgbPoints, alphaPoints);
}

bool HelloMonkey::SetTransferFunc( std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints, unsigned char nLabel)
{
	if (!m_pRender)
		return false;

	return m_pRender->SetTransferFunc(rgbPoints, alphaPoints, nLabel);
}

bool HelloMonkey::LoadTransferFunction(const char* szFile)
{
	if (!m_pRender)
		return false;

	return m_pRender->LoadTransferFunction(szFile);
}

bool HelloMonkey::SaveTransferFunction(const char* szFile)
{
	if (!m_pRender)
		return false;

	return m_pRender->SaveTransferFunction(szFile);
}

void HelloMonkey::SetColorBackground(RGBA clrBG)
{
	if (!m_pRender)
		return;

	m_pRender->SetColorBackground(clrBG);
}

void HelloMonkey::LoadVolumeFile( const char* szFile )
{
	if (!m_pRender)
		return;

	m_pRender->LoadVolumeFile(szFile);
}

void HelloMonkey::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	if (!m_pRender)
		return;
	m_pRender->SetDirection(dirX, dirY, dirZ);
}

void HelloMonkey::SetSpacing( double x, double y, double z )
{
	if (!m_pRender)
		return;
	m_pRender->SetSpacing(x, y, z);
}

void HelloMonkey::SetOrigin(Point3d pt)
{
	if (!m_pRender)
		return;
	m_pRender->SetOrigin(pt);
}

void HelloMonkey::Reset()
{
	if (!m_pRender)
		return;
	m_pRender->Reset();
}

bool HelloMonkey::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	if (!m_pRender)
		return false;
	return m_pRender->SetVolumeData(pData, nWidth, nHeight, nDepth);
}

unsigned char HelloMonkey::AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	if (!m_pRender)
		return 0;
	return m_pRender->AddNewObjectMask(pData, nWidth, nHeight, nDepth);
}

unsigned char HelloMonkey::AddObjectMaskFile(const char* szFile)
{
	if (!m_pRender)
		return 0;
	return m_pRender->AddObjectMaskFile(szFile);
}

bool HelloMonkey::UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	if (!m_pRender)
		return false;
	return m_pRender->UpdateObjectMask(pData, nWidth, nHeight, nDepth, nLabel);
}

std::shared_ptr<short> HelloMonkey::GetVolumeData(int& nWidth, int& nHeight, int& nDepth)
{
	if (!m_pRender)
		return NULL;
	return m_pRender->GetVolumeData(nWidth, nHeight, nDepth);
}

bool HelloMonkey::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	if (!m_pRender)
		return false;
	return m_pRender->GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool HelloMonkey::GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	if (!m_pRender)
		return NULL;
	return m_pRender->GetPlaneData(pData, nWidth, nHeight, planeType);
}

std::string HelloMonkey::GetPlaneData_pngString(const PlaneType& planeType)
{
	if (!m_pRender)
		return "";

	StopWatch sw("GetPlaneData_pngString[%s]", PlaneTypeName(planeType).c_str());

	int nWidth = 0, nHeight = 0;
 	std::shared_ptr<short> pData;
	{
		StopWatch sw("GetPlaneData");
		if (!m_pRender->GetPlaneData(pData, nWidth, nHeight, planeType))
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
	if (!m_pRender)
		return "";

	StopWatch sw("GetOriginData_pngString");

	int nWidth = 0, nHeight = 0, nDepth = 0;
	std::shared_ptr<short> pData = GetVolumeData(nWidth, nHeight, nDepth);
	std::shared_ptr<unsigned char> pMask = m_pRender->GetMaskData();

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

bool HelloMonkey::GetVRData( std::shared_ptr<unsigned char>& pData, int nWidth, int nHeight )
{
	if (!m_pRender)
		return false;
	if (!m_pRender->GetVRData(pData, nWidth, nHeight))
		return false;

	return true;	
}


std::vector<uint8_t> HelloMonkey::GetVRData_png(int nWidth, int nHeight)
{
	StopWatch sw("GetVRData_png");
	std::vector<uint8_t> out_buf;
	if (!m_pRender)
		return out_buf;
	
 	std::shared_ptr<unsigned char> pVR;
	{
		StopWatch sw("GetVRData");
		if (!m_pRender->GetVRData(pVR, nWidth, nHeight))
			return out_buf;

		if (m_bShowCPRLineInVR)
		{
			std::vector<Point3d> cprLine = m_pRender->GetCPRLineVoxel();
			StopWatch sw("cpr line size: %d", cprLine.size());
			if (cprLine.size() >= 2){
				float x0, y0, x1, y1;
				m_pRender->TransferVoxel2ImageInVR(x0, y0, nWidth, nHeight, cprLine[0]);
				for (int i=1; i<cprLine.size(); i++){
					m_pRender->TransferVoxel2ImageInVR(x1, y1, nWidth, nHeight, cprLine[i]);
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
	if (!m_pRender)
		return NULL;
	return m_pRender->SetVRSize(nWidth, nHeight);
}

std::string HelloMonkey::GetVRData_pngString()
{
	StopWatch sw("GetVRData_pngString");

	std::string strBase64 = "";
	int w=-1, h=-1;
	if (!m_pRender){
		return strBase64;
	}
	m_pRender->GetVRSize(w, h);
	std::vector<uint8_t> out_buf = GetVRData_png(w, h);

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
	if (!m_pRender)
		return NULL;
	return m_pRender->GetBatchData(vecBatchData, batchInfo);
}

bool HelloMonkey::GetPlaneIndex( int& index, const PlaneType&  planeType )
{
	if (!m_pRender)
		return NULL;
	return m_pRender->GetPlaneIndex(index, planeType);
}

int HelloMonkey::GetPlaneIndex( const PlaneType&  planeType )
{
	int index = -1;
	if (!m_pRender)
		return index;
	m_pRender->GetPlaneIndex(index, planeType);
	return index;
}

bool HelloMonkey::GetPlaneNumber( int& nTotalNum, const PlaneType&  planeType )
{
	if (!m_pRender)
		return NULL;
	return m_pRender->GetPlaneNumber(nTotalNum, planeType);
}

int HelloMonkey::GetPlaneNumber( const PlaneType&  planeType )
{
	int nTotalNum = -1;
	if (!m_pRender)
		return nTotalNum;
	m_pRender->GetPlaneNumber(nTotalNum, planeType);
	return nTotalNum;
}

bool HelloMonkey::GetPlaneRotateMatrix( float* pMatrix, PlaneType planeType )
{
	if (!m_pRender)
		return NULL;
	return m_pRender->GetPlaneRotateMatrix(pMatrix, planeType);
}

void HelloMonkey::Anterior()
{
	if (!m_pRender)
		return;
	m_pRender->Anterior();
}

void HelloMonkey::Posterior()
{
	if (!m_pRender)
		return;
	m_pRender->Posterior();
}

void HelloMonkey::Left()
{
	if (!m_pRender)
		return;
	m_pRender->Left();
}

void HelloMonkey::Right()
{
	if (!m_pRender)
		return;
	m_pRender->Right();
}

void HelloMonkey::Head()
{
	if (!m_pRender)
		return;
	m_pRender->Head();
}

void HelloMonkey::Foot()
{
	if (!m_pRender)
		return;
	m_pRender->Foot();
}

void HelloMonkey::SetRenderType(RenderType type)
{
	if (!m_pRender)
		return;
	m_pRender->SetRenderType(type);
}

void HelloMonkey::Rotate( float fxRotate, float fyRotate )
{
	if (!m_pRender)
		return;
	m_pRender->Rotate(fxRotate, fyRotate);
}

float HelloMonkey::Zoom( float ratio )
{
	if (!m_pRender)
		return 0.0f;
	return m_pRender->Zoom(ratio);
}

float HelloMonkey::GetZoomRatio()
{
	if (!m_pRender)
		return 0.0f;
	return m_pRender->GetZoomRatio();
}

void HelloMonkey::Pan( float fxShift, float fyShift )
{
	if (!m_pRender)
		return;
	m_pRender->Pan(fxShift, fyShift);
}

bool HelloMonkey::SetVRWWWL(float fWW, float fWL)
{
	if (!m_pRender)
		return false;
	return m_pRender->SetVRWWWL(fWW, fWL);
}

bool HelloMonkey::SetVRWWWL(float fWW, float fWL, unsigned char nLabel)
{
	if (!m_pRender)
		return false;
	return m_pRender->SetVRWWWL(fWW, fWL, nLabel);
}

bool HelloMonkey::SetObjectAlpha(float fAlpha)
{
	if (!m_pRender)
		return false;
	return m_pRender->SetObjectAlpha(fAlpha);
}

bool HelloMonkey::SetObjectAlpha(float fAlpha, unsigned char nLabel)
{
	if (!m_pRender)
		return false;
	return m_pRender->SetObjectAlpha(fAlpha, nLabel);
}

void HelloMonkey::Browse( float fDelta, PlaneType planeType )
{
	if (!m_pRender)
		return;
	m_pRender->Browse(fDelta, planeType);
}

void HelloMonkey::PanCrossHair( float fx, float fy, PlaneType planeType )
{
	if (!m_pRender)
		return;
	m_pRender->PanCrossHair(fx, fy, planeType);
}

bool HelloMonkey::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	if (!m_pRender)
		return false;
	return m_pRender->GetCrossHairPoint(x, y, planeType);
}

Point2d HelloMonkey::GetCrossHairPoint(const PlaneType& planeType)
{
	Point2d pt(-1, -1);

	if (!m_pRender)
		return pt;
	double x = -1, y = -1;
	if (m_pRender->GetCrossHairPoint(x, y, planeType)){
		pt = Point2d(x, y);
	}
	return pt;
}

bool HelloMonkey::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
{
	if (!m_pRender)
		return false;
	return m_pRender->GetDirection(dirH, dirV, planeType);
}

bool HelloMonkey::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	if (!m_pRender)
		return false;
	return m_pRender->GetDirection3D(dir3dH, dir3dV, planeType);
}

bool HelloMonkey::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType )
{
	if (!m_pRender)
		return false;
	return m_pRender->GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void HelloMonkey::RotateCrossHair( float fAngle, PlaneType planeType )
{
	if (!m_pRender)
		return;
	m_pRender->RotateCrossHair(fAngle, planeType);
}

void HelloMonkey::SetPlaneIndex( int index, PlaneType planeType )
{
	if (!m_pRender)
		return;
	m_pRender->SetPlaneIndex(index, planeType);
}

double HelloMonkey::GetPixelSpacing( PlaneType planeType )
{
	if (!m_pRender)
		return 1.0;
	return m_pRender->GetPixelSpacing(planeType);
}

bool HelloMonkey::TransferImage2Voxel( double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType )
{
	if (!m_pRender)
		return false;
	return m_pRender->TransferImage2Voxel( x, y, z, xImage, yImage, planeType );
}

bool HelloMonkey::GetCrossHairPoint3D( Point3d& pt )
{
	if (!m_pRender)
		return false;
	return m_pRender->GetCrossHairPoint3D( pt );
}

void HelloMonkey::UpdateThickness( double val )
{
	if (!m_pRender)
		return;
	return m_pRender->UpdateThickness( val );
}

void HelloMonkey::SetThickness(double val, PlaneType planeType)
{
	if (!m_pRender)
		return;
	return m_pRender->SetThickness(val, planeType);
}

double HelloMonkey::GetThickness(PlaneType planeType)
{
	if (!m_pRender)
		return false;
	return m_pRender->GetThickness(planeType);
}

void HelloMonkey::SetMPRType( MPRType type )
{
	if (!m_pRender)
		return;
	return m_pRender->SetMPRType(type);
}

bool HelloMonkey::SetCPRLinePatient(std::vector<Point3d> cprLine)
{
	if (!m_pRender)
		return false;
	return m_pRender->SetCPRLinePatient(cprLine);
}

bool HelloMonkey::SetCPRLineVoxel(std::vector<Point3d> cprLine)
{
	if (!m_pRender)
		return false;
	return m_pRender->SetCPRLineVoxel(cprLine);
}

bool HelloMonkey::RotateCPR(float angle, PlaneType planeType)
{
	if (!m_pRender)
		return false;
	return m_pRender->RotateCPR(angle, planeType);
}

void HelloMonkey::ShowCPRLineInVR(bool bShow)
{
	m_bShowCPRLineInVR = bShow;
}

void HelloMonkey::ShowPlaneInVR(bool bShow)
{
	if (!m_pRender)
		return;
	return m_pRender->ShowPlaneInVR(bShow);
}
