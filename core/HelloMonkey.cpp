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

using namespace MonkeyGL;

IRender* _pRender = NULL;

HelloMonkey::HelloMonkey()
{
	if (NULL != _pRender)
	{
		delete _pRender;
		_pRender = NULL;
	}

	_pRender = new Render();

	Logger::Init();

	Logger::Info("MonkeyGL has started....");

	fpng::fpng_init();
	if (fpng::fpng_cpu_supports_sse41()){
		Logger::Info("fpng cpu supports sse41");
	}
	else {
		Logger::Error("fpng cpu not supports sse41");
	}
}

HelloMonkey::~HelloMonkey(void)
{
	if (NULL != _pRender)
	{
		delete _pRender;
		_pRender = NULL;
	}
}

void HelloMonkey::SetLogLevel(LogLevel level){
	Logger::SetLevel(level);
}

void HelloMonkey::SetTransferFunc( const std::map<int, RGBA>& ctrlPoints )
{
	if (NULL == _pRender)
		return;
	
	_pRender->SetTransferFunc(ctrlPoints);
}

void HelloMonkey::SetTransferFunc( const std::map<int, RGBA>& rgbPoints, const std::map<int, double>& alphaPoints)
{
	if (NULL == _pRender)
		return;

	_pRender->SetTransferFunc(rgbPoints, alphaPoints);
}

void HelloMonkey::SetVolumeFile( const char* szFile, int nWidth, int nHeight, int nDepth )
{
	if (NULL == _pRender)
		return;

	_pRender->SetVolumeFile(szFile, nWidth, nHeight, nDepth);
}

void HelloMonkey::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	if (NULL == _pRender)
		return;
	_pRender->SetDirection(dirX, dirY, dirZ);
}

void HelloMonkey::SetAnisotropy( double x, double y, double z )
{
	if (NULL == _pRender)
		return;
	_pRender->SetAnisotropy(x, y, z);
}

void HelloMonkey::Reset()
{
	if (NULL == _pRender)
		return;
	_pRender->Reset();
}

short* HelloMonkey::GetVolumeData()
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetVolumeData();
}

bool HelloMonkey::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool HelloMonkey::GetPlaneData(short* pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneData(pData, nWidth, nHeight, planeType);
}

std::string HelloMonkey::GetPlaneData_pngString(const PlaneType& planeType)
{
	if (NULL == _pRender)
		return "";

	StopWatch sw("GetPlaneData_pngString");

	int nWidth = 0, nHeight = 0;
	GetPlaneMaxSize(nWidth, nHeight, planeType);
	
 	std::shared_ptr<short> pData (new short[nWidth*nHeight]);
	{
		StopWatch sw("GetPlaneData");
		if (!_pRender->GetPlaneData(pData.get(), nWidth, nHeight, planeType))
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
			Logger::FormatMsg(
				"plane encode, from %d to %d, ratio %.4f",
				nWidth*nHeight*sizeof(short),
				out_buf.size(),
				1.0*out_buf.size()/(nWidth*nHeight*sizeof(short))
			)
		);
	}

	std::string strBase64 = "";
	{
		StopWatch sw("Base64 Encode");
		strBase64 = Base64::Encode(out_buf.data(), out_buf.size());

		Logger::Info(
			Logger::FormatMsg(
				"plane base64, from %d to %d, ratio %.4f",
				out_buf.size(),
				strBase64.length(),
				1.0*strBase64.length()/out_buf.size()
			)
		);
	}

	return strBase64;
}

bool HelloMonkey::GetVRData( unsigned char* pVR, int nWidth, int nHeight )
{
	if (NULL == _pRender)
		return false;
	if (!_pRender->GetVRData(pVR, nWidth, nHeight))
		return false;

	return true;	
}


std::vector<uint8_t> HelloMonkey::GetVRData_png(int nWidth, int nHeight)
{
	StopWatch sw("GetVRData_png");
	std::vector<uint8_t> out_buf;
	if (NULL == _pRender)
		return out_buf;
	
 	std::shared_ptr<unsigned char> pVR (new unsigned char[nWidth*nHeight*3]);

	{
		StopWatch sw("GetVRData");
		if (!_pRender->GetVRData(pVR.get(), nWidth, nHeight))
			return out_buf;
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
			Logger::FormatMsg(
				"vr encode, from %d to %d, ratio %.4f",
				nWidth*nHeight*3,
				out_buf.size(),
				1.0*out_buf.size()/(nWidth*nHeight*3)
			)
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
		Logger::Error(
			Logger::FormatMsg(
				"failed to save png file [%s]",
				szFile
			)
		);
	}
	fwrite(out_buf.data(), 1, nWidth*nHeight*3, fp);
	fclose(fp);
	Logger::Info(
		Logger::FormatMsg(
			"saved png file [%s]",
			szFile
		)
	);
}

std::string HelloMonkey::GetVRData_pngString(int nWidth, int nHeight)
{
	StopWatch sw("GetVRData_pngString");
	std::vector<uint8_t> out_buf = GetVRData_png(nWidth, nHeight);

	std::string strBase64 = "";
	{
		StopWatch sw("Base64 Encode");
		strBase64 = Base64::Encode(out_buf.data(), out_buf.size());

		Logger::Info(
			Logger::FormatMsg(
				"vr base64, from %d to %d, ratio %.4f",
				out_buf.size(),
				strBase64.length(),
				1.0*strBase64.length()/out_buf.size()
			)
		);
	}

	return strBase64;
}

bool HelloMonkey::GetBatchData( std::vector<short*>& vecBatchData, const BatchInfo& batchInfo )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetBatchData(vecBatchData, batchInfo);
}

bool HelloMonkey::GetPlaneIndex( int& index, PlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneIndex(index, planeType);
}

bool HelloMonkey::GetPlaneNumber( int& nTotalNum, PlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneNumber(nTotalNum, planeType);
}

bool HelloMonkey::GetPlaneRotateMatrix( float* pMatrix, PlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneRotateMatrix(pMatrix, planeType);
}

void HelloMonkey::Anterior()
{
	if (NULL == _pRender)
		return;
	_pRender->Anterior();
}

void HelloMonkey::Posterior()
{
	if (NULL == _pRender)
		return;
	_pRender->Posterior();
}

void HelloMonkey::Left()
{
	if (NULL == _pRender)
		return;
	_pRender->Left();
}

void HelloMonkey::Right()
{
	if (NULL == _pRender)
		return;
	_pRender->Right();
}

void HelloMonkey::Head()
{
	if (NULL == _pRender)
		return;
	_pRender->Head();
}

void HelloMonkey::Foot()
{
	if (NULL == _pRender)
		return;
	_pRender->Foot();
}

void HelloMonkey::Rotate( float fxRotate, float fyRotate )
{
	if (NULL == _pRender)
		return;
	_pRender->Rotate(fxRotate, fyRotate);
}

void HelloMonkey::Zoom( float ratio )
{
	if (NULL == _pRender)
		return;
	_pRender->Zoom(ratio);
}

void HelloMonkey::Pan( float fxShift, float fyShift )
{
	if (NULL == _pRender)
		return;
	_pRender->Pan(fxShift, fyShift);
}

void HelloMonkey::SetVRWWWL(float fWW, float fWL)
{
	if (NULL == _pRender)
		return;
	_pRender->SetVRWWWL(fWW, fWL);
}

void HelloMonkey::Browse( float fDelta, PlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->Browse(fDelta, planeType);
}

void HelloMonkey::PanCrossHair( int nx, int ny, PlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->PanCrossHair(nx, ny, planeType);
}

bool HelloMonkey::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetCrossHairPoint(x, y, planeType);
}

bool HelloMonkey::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetDirection(dirH, dirV, planeType);
}

bool HelloMonkey::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetDirection3D(dir3dH, dir3dV, planeType);
}

bool HelloMonkey::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void HelloMonkey::RotateCrossHair( float fAngle, PlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->RotateCrossHair(fAngle, planeType);
}

void HelloMonkey::SetPlaneIndex( int index, PlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->SetPlaneIndex(index, planeType);
}

double HelloMonkey::GetPixelSpacing( PlaneType planeType )
{
	if (NULL == _pRender)
		return 1.0;
	return _pRender->GetPixelSpacing(planeType);
}

bool HelloMonkey::TransferImage2Object( double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->TransferImage2Object( x, y, z, xImage, yImage, planeType );
}

bool HelloMonkey::GetCrossHairPoint3D( Point3d& pt )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetCrossHairPoint3D( pt );
}

void HelloMonkey::UpdateThickness( double val )
{
	if (NULL == _pRender)
		return;
	return _pRender->UpdateThickness( val );
}

void HelloMonkey::SetThickness(double val, PlaneType planeType)
{
	if (NULL == _pRender)
		return;
	return _pRender->SetThickness(val, planeType);
}

bool HelloMonkey::GetThickness(double& val, PlaneType planeType)
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetThickness(val, planeType);
}

void HelloMonkey::SetMPRType( MPRType type )
{
	if (NULL == _pRender)
		return;
	return _pRender->SetMPRType(type);
}
