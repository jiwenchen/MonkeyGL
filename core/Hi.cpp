#include "Hi.h"
#include "Render.h"

using namespace MonkeyGL;

IRender* _pRender = NULL;

Hi::Hi()
{
	if (NULL != _pRender)
	{
		delete _pRender;
		_pRender = NULL;
	}

	_pRender = new Render();
}

Hi::~Hi(void)
{
	if (NULL != _pRender)
	{
		delete _pRender;
		_pRender = NULL;
	}
}

void Hi::SetTransferFunc( const std::map<int, RGBA>& ctrlPoints )
{
	if (NULL == _pRender)
		return;
	
	_pRender->SetTransferFunc(ctrlPoints);
}

void Hi::SetTransferFunc( const std::map<int, RGBA>& rgbPoints, const std::map<int, double>& alphaPoints)
{
	if (NULL == _pRender)
		return;

	_pRender->SetTransferFunc(rgbPoints, alphaPoints);
}

void Hi::SetVolumeFile( const char* szFile, int nWidth, int nHeight, int nDepth )
{
	if (NULL == _pRender)
		return;

	_pRender->SetVolumeFile(szFile, nWidth, nHeight, nDepth);
}

void Hi::SetDirection( Direction3d dirX, Direction3d dirY, Direction3d dirZ )
{
	if (NULL == _pRender)
		return;
	_pRender->SetDirection(dirX, dirY, dirZ);
}

void Hi::SetAnisotropy( double x, double y, double z )
{
	if (NULL == _pRender)
		return;
	_pRender->SetAnisotropy(x, y, z);
}

void Hi::Reset()
{
	if (NULL == _pRender)
		return;
	_pRender->Reset();
}

bool Hi::GetPlaneMaxSize( int& nWidth, int& nHeight, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool Hi::GetPlaneData(short* pData, int& nWidth, int& nHeight, const ePlaneType& planeType)
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneData(pData, nWidth, nHeight, planeType);
}

bool Hi::GetVRData( unsigned char* pVR, int nWidth, int nHeight )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetVRData(pVR, nWidth, nHeight);
}

bool Hi::GetBatchData( std::vector<short*>& vecBatchData, const BatchInfo& batchInfo )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetBatchData(vecBatchData, batchInfo);
}

bool Hi::GetPlaneIndex( int& index, ePlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneIndex(index, planeType);
}

bool Hi::GetPlaneNumber( int& nTotalNum, ePlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneNumber(nTotalNum, planeType);
}

bool Hi::GetPlaneRotateMatrix( float* pMatrix, ePlaneType planeType )
{
	if (NULL == _pRender)
		return NULL;
	return _pRender->GetPlaneRotateMatrix(pMatrix, planeType);
}

void Hi::Anterior()
{
	if (NULL == _pRender)
		return;
	_pRender->Anterior();
}

void Hi::Posterior()
{
	if (NULL == _pRender)
		return;
	_pRender->Posterior();
}

void Hi::Left()
{
	if (NULL == _pRender)
		return;
	_pRender->Left();
}

void Hi::Right()
{
	if (NULL == _pRender)
		return;
	_pRender->Right();
}

void Hi::Head()
{
	if (NULL == _pRender)
		return;
	_pRender->Head();
}

void Hi::Foot()
{
	if (NULL == _pRender)
		return;
	_pRender->Foot();
}

void Hi::Rotate( float fxRotate, float fyRotate )
{
	if (NULL == _pRender)
		return;
	_pRender->Rotate(fxRotate, fyRotate);
}

void Hi::Zoom( float ratio )
{
	if (NULL == _pRender)
		return;
	_pRender->Zoom(ratio);
}

void Hi::Pan( float fxShift, float fyShift )
{
	if (NULL == _pRender)
		return;
	_pRender->Pan(fxShift, fyShift);
}

void Hi::SetWL(float fWW, float fWL)
{
	if (NULL == _pRender)
		return;
	_pRender->SetWL(fWW, fWL);
}

void Hi::Browse( float fDelta, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->Browse(fDelta, planeType);
}

void Hi::PanCrossHair( int nx, int ny, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->PanCrossHair(nx, ny, planeType);
}

bool Hi::GetCrossHairPoint( double& x, double& y, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetCrossHairPoint(x, y, planeType);
}

bool Hi::GetDirection( Direction2d& dirH, Direction2d& dirV, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetDirection(dirH, dirV, planeType);
}

bool Hi::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetDirection3D(dir3dH, dir3dV, planeType);
}

bool Hi::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const ePlaneType& planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void Hi::RotateCrossHair( float fAngle, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->RotateCrossHair(fAngle, planeType);
}

void Hi::SetPlaneIndex( int index, ePlaneType planeType )
{
	if (NULL == _pRender)
		return;
	_pRender->SetPlaneIndex(index, planeType);
}

double Hi::GetPixelSpacing( ePlaneType planeType )
{
	if (NULL == _pRender)
		return 1.0;
	return _pRender->GetPixelSpacing(planeType);
}

bool Hi::TransferImage2Object( double& x, double& y, double& z, double xImage, double yImage, ePlaneType planeType )
{
	if (NULL == _pRender)
		return false;
	return _pRender->TransferImage2Object( x, y, z, xImage, yImage, planeType );
}

bool Hi::GetCrossHairPoint3D( Point3d& pt )
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetCrossHairPoint3D( pt );
}

void Hi::UpdateThickness( double val )
{
	if (NULL == _pRender)
		return;
	return _pRender->UpdateThickness( val );
}

void Hi::SetThickness(double val, ePlaneType planeType)
{
	if (NULL == _pRender)
		return;
	return _pRender->SetThickness(val, planeType);
}

bool Hi::GetThickness(double& val, ePlaneType planeType)
{
	if (NULL == _pRender)
		return false;
	return _pRender->GetThickness(val, planeType);
}

void Hi::SetMPRType( MPRType type )
{
	if (NULL == _pRender)
		return;
	return _pRender->SetMPRType(type);
}
