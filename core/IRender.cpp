#include "IRender.h"

using namespace MonkeyGL;

IRender::IRender(void)
{
}


IRender::~IRender(void)
{
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

short* IRender::GetVolumeData()
{
	return m_dataMan.GetVolumeData();
}

bool IRender::GetPlaneMaxSize( int& nWidth, int& nHeight, const ePlaneType& planeType )
{
	return m_dataMan.GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool IRender::GetPlaneData(short* pData, int& nWidth, int& nHeight, const ePlaneType& planeType)
{
	return false;
}

bool IRender::GetCrossHairPoint( double& x, double& y, const ePlaneType& planeType )
{
	return m_dataMan.GetCrossHairPoint(x, y, planeType);
}

bool IRender::TransferImage2Object(double& x, double& y, double& z, double xImage, double yImage, ePlaneType planeType)
{
	return m_dataMan.TransferImage2Object(x, y, z, xImage, yImage, planeType);
}

bool IRender::GetCrossHairPoint3D( Point3d& pt )
{
	pt = m_dataMan.GetCrossHair();
	return true;
}

bool IRender::GetDirection( Direction2d& dirH, Direction2d& dirV, const ePlaneType& planeType )
{
	return m_dataMan.GetDirection(dirH, dirV, planeType);
}

bool IRender::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const ePlaneType& planeType )
{
	return m_dataMan.GetDirection3D(dir3dH, dir3dV, planeType);
}

bool IRender::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const ePlaneType& planeType )
{
	return m_dataMan.GetBatchDirection3D(dir3dH, dir3dV, fAngle, planeType);
}

void IRender::Browse( float fDelta, ePlaneType planeType )
{
	m_dataMan.Browse(fDelta, planeType);
}

void IRender::SetPlaneIndex( int index, ePlaneType planeType )
{
	m_dataMan.SetPlaneIndex(index, planeType);
}

void IRender::PanCrossHair( int nx, int ny, ePlaneType planeType )
{
	m_dataMan.PanCrossHair(nx, ny, planeType);
}

void IRender::RotateCrossHair( float fAngle, ePlaneType planeType )
{
	m_dataMan.RotateCrossHair(fAngle, planeType);
}

double IRender::GetPixelSpacing( ePlaneType planeType )
{
	return m_dataMan.GetPixelSpacing(planeType);
}

bool IRender::GetPlaneIndex( int& index, ePlaneType planeType )
{
	return m_dataMan.GetPlaneIndex(index, planeType);
}

bool IRender::GetPlaneNumber( int& nTotalNum, ePlaneType planeType )
{
	return m_dataMan.GetPlaneNumber(nTotalNum, planeType);
}

bool IRender::GetPlaneRotateMatrix( float* pMatirx, ePlaneType planeType )
{
	return m_dataMan.GetPlaneRotateMatrix(pMatirx, planeType);
}

void IRender::UpdateThickness( double val )
{
	m_dataMan.UpdateThickness(val);
}

void IRender::SetThickness(double val, ePlaneType planeType)
{
	m_dataMan.SetThickness(val, planeType);
}

bool IRender::GetThickness(double& val, ePlaneType planeType)
{
	return m_dataMan.GetThickness(val, planeType);
}

void IRender::SetMPRType(MPRType type)
{
	m_dataMan.SetMPRType(type);
}
