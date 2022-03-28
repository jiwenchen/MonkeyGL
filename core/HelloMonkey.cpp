#include "HelloMonkey.h"
#include "Render.h"

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
}

HelloMonkey::~HelloMonkey(void)
{
	if (NULL != _pRender)
	{
		delete _pRender;
		_pRender = NULL;
	}
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

bool HelloMonkey::GetPlaneMaxSize( int& nWidth, int& nHeight, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool HelloMonkey::GetPlaneData(short* pData, int& nWidth, int& nHeight, const ePlaneType& planeType)
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneData(pData, nWidth, nHeight, planeType);
}

bool HelloMonkey::GetVRData( unsigned char* pVR, int nWidth, int nHeight )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetVRData(pVR, nWidth, nHeight);
}

void HelloMonkey::SaveVR2BMP(const char* szFile, int nWidth, int nHeight)
{
	if (NULL == _pRender)
		return;
	return _pRender->SaveVR2BMP(szFile, nWidth, nHeight);
}

bool HelloMonkey::GetBatchData( std::vector<short*>& vecBatchData, const BatchInfo& batchInfo )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetBatchData(vecBatchData, batchInfo);
}

bool HelloMonkey::GetPlaneIndex( int& index, ePlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneIndex(index, planeType);
}

bool HelloMonkey::GetPlaneNumber( int& nTotalNum, ePlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneNumber(nTotalNum, planeType);
}

bool HelloMonkey::GetPlaneRotateMatrix( float* pMatrix, ePlaneType planeType )
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

void HelloMonkey::SetWL(float fWW, float fWL)
{
	if (NULL == _pRender)
		return;
	_pRender->SetWL(fWW, fWL);
}

void HelloMonkey::Browse( float fDelta, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->Browse(fDelta, planeType);
}

void HelloMonkey::PanCrossHair( int nx, int ny, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->PanCrossHair(nx, ny, planeType);
}

bool HelloMonkey::GetCrossHairPoint( double& x, double& y, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetCrossHairPoint(x, y, planeType);
}

bool HelloMonkey::GetDirection( Direction2d& dirH, Direction2d& dirV, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetDirection(dirH, dirV, planeType);
}

bool HelloMonkey::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetDirection3D(dir3dH, dir3dV, planeType);
}

bool HelloMonkey::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void HelloMonkey::RotateCrossHair( float fAngle, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->RotateCrossHair(fAngle, planeType);
}

void HelloMonkey::SetPlaneIndex( int index, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->SetPlaneIndex(index, planeType);
}

double HelloMonkey::GetPixelSpacing( ePlaneType planeType )
{
	if (NULL == _pRender)
		return 1.0;
	return _pRender->GetPixelSpacing(planeType);
}

bool HelloMonkey::TransferImage2Object( double& x, double& y, double& z, double xImage, double yImage, ePlaneType planeType )
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

void HelloMonkey::SetThickness(double val, ePlaneType planeType)
{
	if (NULL == _pRender)
		return;
	return _pRender->SetThickness(val, planeType);
}

bool HelloMonkey::GetThickness(double& val, ePlaneType planeType)
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
