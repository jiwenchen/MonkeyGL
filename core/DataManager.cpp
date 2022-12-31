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

#include "DataManager.h"
#include "Logger.h"
#include "Methods.h"
#include "ImageReader.h"
#include "DeviceInfo.h"

using namespace MonkeyGL;

DataManager::DataManager(void)
{
	m_activeLabel = -1;
	m_gpuEnabeld = false;
	TryToEnableGPU(true);
	m_objectInfos.clear();
	
	m_renderType = RenderTypeVR;
}

DataManager::~DataManager(void)
{
	m_objectInfos.clear();
}

DataManager* DataManager::Instance()
{
	static DataManager* pDataManager = NULL;
	if (NULL == pDataManager)
	{
		pDataManager = new DataManager();
	}
	return pDataManager;
}

VolumeInfo& DataManager::GetVolumeInfo()
{
	return m_volInfo;
}

MPRInfo& DataManager::GetMPRInfo()
{
	return m_mprInfo;
}

CPRInfo& DataManager::GetCPRInfo()
{
	return m_cprInfo;
}

RenderInfo& DataManager::GetRenderInfo()
{
	return m_renderInfo;
}

CuDataManager& DataManager::GetCuDataManager()
{
	return m_cuDataMan;
}

AnnotationInfo& DataManager::GetAnnotationInfo()
{
	return m_annotationInfo;
}

bool DataManager::TryToEnableGPU(bool enable)
{
	if (enable && DeviceInfo::Instance()->Initialized()){
		m_gpuEnabeld = true;
	}
	else{
		m_gpuEnabeld = false;
	}
	return m_gpuEnabeld;
}

bool DataManager::GPUEnabled()
{
	return m_gpuEnabeld;
}

void DataManager::ClearAndReset()
{
	m_activeLabel = -1;
	m_objectInfos.clear();
	m_volInfo.Clear();
}

bool DataManager::SetVRSize(int nWidth, int nHeight)
{
	return m_renderInfo.SetVRSize(nWidth, nHeight);
}

void DataManager::GetVRSize(int& nWidth, int& nHeight)
{
	m_renderInfo.GetVRSize(nWidth, nHeight);
}

bool DataManager::LoadVolumeFile( const char* szFile )
{
	ClearAndReset();
	bool res = m_volInfo.LoadVolumeFile(szFile);
	m_cprInfo.SetSpacing(Point3d(GetSpacing(0), GetSpacing(1), GetSpacing(2)));
	m_mprInfo.ResetPlaneInfos();
	m_activeLabel = 0;
	m_objectInfos[0] = ObjectInfo();

	m_VolumeSize.width = GetDim(0);
	m_VolumeSize.height = GetDim(1);
	m_VolumeSize.depth = GetDim(2);

	NormalizeVOI();

	if (DeviceInfo::Instance()->Initialized()){

		m_cuDataMan.CopyVolumeData(GetVolumeData().get(), m_VolumeSize);
		InitCommon(GetSpacing(0), GetSpacing(1), GetSpacing(2), m_VolumeSize);
		m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
	}

	return res;
}

bool DataManager::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	ClearAndReset();
	if (!m_volInfo.SetVolumeData(pData, nWidth, nHeight, nDepth))
		return false;
	m_mprInfo.ResetPlaneInfos();
	m_activeLabel = 0;
	m_objectInfos[0] = ObjectInfo();

	m_VolumeSize.width = GetDim(0);
	m_VolumeSize.height = GetDim(1);
	m_VolumeSize.depth = GetDim(2);

	NormalizeVOI();

	if (DeviceInfo::Instance()->Initialized()){
		m_cuDataMan.CopyVolumeData(GetVolumeData().get(), m_VolumeSize);
		m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
	}

	return true;
}

void DataManager::NormalizeVOI()
{
	m_voiNormalize.left = 0;
	m_voiNormalize.right = m_VolumeSize.width - 1;
	m_voiNormalize.posterior = 0;
	m_voiNormalize.anterior = m_VolumeSize.height - 1;
	m_voiNormalize.head = 0;
	m_voiNormalize.foot = m_VolumeSize.depth - 1;
}

void DataManager::Rotate(float fxRotate, float fyRotate)
{
	m_renderInfo.Rotate(fxRotate, fyRotate);
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
}

float DataManager::Zoom(float ratio)
{
	float newRatio = m_renderInfo.Zoom(ratio);
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
	return newRatio;
}

float DataManager::GetZoomRatio()
{
	return m_renderInfo.GetZoomRatio();
}

void DataManager::Pan(float fxShift, float fyShift)
{
	m_renderInfo.Pan(fxShift, fyShift);
}

void DataManager::Anterior()
{
	m_renderInfo.Anterior();
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
}

void DataManager::Posterior()
{
	m_renderInfo.Posterior();
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
}

void DataManager::Left()
{
	m_renderInfo.Left();
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
}

void DataManager::Right()
{
	m_renderInfo.Right();
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
}

void DataManager::Head()
{
	m_renderInfo.Head();
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
}

void DataManager::Foot()
{
	m_renderInfo.Foot();
	m_cuDataMan.CopyOperatorMatrix(m_renderInfo.m_pTransformMatrix, m_renderInfo.m_pTransposeTransformMatrix);
}

bool DataManager::TransferVoxel2ImageInVR(float &fx, float &fy, int nWidth, int nHeight, Point3d ptVoxel)
{
	double fxSpacing = GetSpacing(0);
	double fySpacing = GetSpacing(1);
	double fzSpacing = GetSpacing(2);

	float fMaxLen = max(m_VolumeSize.width * fxSpacing, max(m_VolumeSize.height * fySpacing, m_VolumeSize.depth * fzSpacing));
	Point3d maxper(fMaxLen / (m_VolumeSize.width * fxSpacing), fMaxLen / (m_VolumeSize.height * fySpacing), fMaxLen / (m_VolumeSize.depth * fzSpacing));

	Point3d pt(ptVoxel.x() / m_VolumeSize.width, ptVoxel.y() / m_VolumeSize.height, ptVoxel.z() / m_VolumeSize.depth);
	if (Need2InvertZ())
	{
		pt.SetZ(1.0 - pt.z());
	}
	pt -= Point3d(0.5, 0.5, 0.5);
	pt /= maxper;

	pt = Methods::MatrixMul(m_renderInfo.m_pTransposeTransformMatrix, pt);
	fx = (pt.x() + 0.5) * nWidth + m_renderInfo.m_fTotalXTranslate;
	fy = (pt.z() + 0.5) * nHeight + m_renderInfo.m_fTotalYTranslate;

	return true;
}

unsigned char DataManager::AddObjectMaskFile(const char* szFile)
{
	std::shared_ptr<unsigned char> pData;
	int nWidth=0, nHeight=0, nDepth=0;
	if (!ImageReader::ReadMask(
			szFile,
			pData,
			nWidth,
			nHeight,
			nDepth
		)
	){
		Logger::Error("failed to read mask file[%s]", szFile);
		return -1;
	}

	return AddNewObjectMask(pData, nWidth, nHeight, nDepth);
}


unsigned char DataManager::AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	Logger::Info("add new mask, size[w:%d, h:%d, d:%d]", nWidth, nHeight, nDepth);
	unsigned char nLabel = 0;
	if (m_objectInfos.size() <= 0) {
		m_objectInfos[0] = ObjectInfo();
		nLabel = 1;
	}
	else if (m_objectInfos.size() >= MAXOBJECTCOUNT+1) {
		Logger::Warn("failed to add new mask, since the labels is full.");
		return 0;
	}
	else {
		int idx = 1;
		for (std::map<unsigned char, ObjectInfo>::iterator iter=++m_objectInfos.begin(); iter!=m_objectInfos.end(); iter++){
			if (idx != iter->first){
				break;
			}
			else{
				idx++;
			}
		}
		nLabel = idx;
	}
	if(!m_volInfo.AddNewObjectMask(pData, nWidth, nHeight, nDepth, nLabel)){
		return 0;
	}

	m_objectInfos[nLabel] = m_objectInfos[m_activeLabel];
	m_activeLabel = nLabel;

	if (GPUEnabled()){
		m_cuDataMan.CopyMaskData(GetMaskData().get(), m_VolumeSize);
	}
	return nLabel;
}

bool DataManager::UpdateActiveObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	if (m_activeLabel < 0 || m_activeLabel > MAXOBJECTCOUNT)
		return false;
	return UpdateObjectMask(pData, nWidth, nHeight, nDepth, m_activeLabel);
}

bool DataManager::UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	if (m_objectInfos.find(nLabel) == m_objectInfos.end()){
		return false;
	}
	bool res = m_volInfo.UpdateObjectMask(pData, nWidth, nHeight, nDepth, nLabel);

	if (GPUEnabled()){
		m_cuDataMan.CopyMaskData(GetMaskData().get(), m_VolumeSize);
	}
	return res;
}

bool DataManager::SetControlPoints_TF( std::map<int, RGBA> ctrlPts)
{
	return SetControlPoints_TF(ctrlPts, m_activeLabel);
}

bool DataManager::SetControlPoints_TF( std::map<int, RGBA> ctrlPts, unsigned char nLabel )
{
	if (nLabel < 0 || nLabel > MAXOBJECTCOUNT || m_objectInfos.find(nLabel) == m_objectInfos.end()){
		Logger::Error("invalid lable, maybe there is no mask, please add first.");
		return false;
	}
	m_objectInfos[nLabel].idx2rgba = ctrlPts;

	UpdateTransferFunctions();
	return true;
}

bool DataManager::SetControlPoints_TF( std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts )
{
	return SetControlPoints_TF(rgbPts, alphaPts, m_activeLabel);
}

bool DataManager::SetControlPoints_TF( std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts, unsigned char nLabel )
{
	if (nLabel < 0 || nLabel > MAXOBJECTCOUNT || m_objectInfos.find(nLabel) == m_objectInfos.end()){
		Logger::Error("invalid lable, maybe there is no mask, please add first.");
		return false;
	}
	m_objectInfos[nLabel].idx2rgba = rgbPts;
	m_objectInfos[nLabel].idx2alpha = alphaPts;

	UpdateTransferFunctions();
	return true;
}

void DataManager::UpdateTransferFunctions()
{
	if (!GPUEnabled()){
		return;
	}

	std::shared_ptr<RGBA> ptfBuffer(NULL);
	int ntfLength = 0;

	for (std::map<unsigned char, ObjectInfo>::iterator iter = m_objectInfos.begin(); iter != m_objectInfos.end(); iter++)
	{
		if (iter->second.GetTransferFunction(ptfBuffer, ntfLength))
		{
			m_cuDataMan.SetTransferFunction((float *)(ptfBuffer.get()), ntfLength, iter->first);
		}
	}
}

bool DataManager::LoadTransferFunction(const char* szFile) 
{
	ObjectInfo info;
	ObjectInfo::ReadFile(szFile, info);
	info.Print();
	
	return (
		SetVRWWWL(info.ww, info.wl) 
		&& SetControlPoints_TF(info.idx2rgba, info.idx2alpha)
	);
}

bool DataManager::SaveTransferFunction(const char* szFile) 
{
	if (m_activeLabel < 0){
		return false;
	}
	if (m_objectInfos.size() <= 0){
		return false;
	}

	ObjectInfo::WriteFile(szFile, m_objectInfos[m_activeLabel]);
	return true;
}

std::map<unsigned char, ObjectInfo> DataManager::GetObjectInfos()
{
	return m_objectInfos;
}

void DataManager::SetColorBackground(RGBA clrBG)
{
	m_colorBG = clrBG;
}

RGBA DataManager::GetColorBackground()
{
	return m_colorBG;
}

void DataManager::SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ)
{
	m_volInfo.SetDirection(dirX, dirY, dirZ);
}

void DataManager::SetSpacing( double x, double y, double z )
{
	m_volInfo.SetSpacing(x, y, z);
	m_cprInfo.SetSpacing(Point3d(x, y, z));
	m_mprInfo.ResetPlaneInfos();

	InitCommon(x, y, z, m_VolumeSize);
}

void DataManager::SetOrigin(Point3d pt)
{
	m_volInfo.SetOrigin(pt);
}

void DataManager::Reset()
{
	m_mprInfo.ResetPlaneInfos();
}

std::shared_ptr<short> DataManager::GetVolumeData()
{
	return m_volInfo.GetVolumeData();
}

std::shared_ptr<short> DataManager::GetVolumeData(int& nWidth, int& nHeight, int& nDepth)
{
	return m_volInfo.GetVolumeData(nWidth, nHeight, nDepth);
}

std::shared_ptr<unsigned char> DataManager::GetMaskData()
{
	return m_volInfo.GetMaskData();
}

bool DataManager::SetCPRLinePatient(std::vector<Point3d> cprLine)
{
	return m_cprInfo.SetCPRLinePatient(cprLine);
}

bool DataManager::SetCPRLineVoxel(std::vector<Point3d> cprLine)
{
	return m_cprInfo.SetCPRLineVoxel(cprLine);
}

std::vector<Point3d> DataManager::GetCPRLineVoxel()
{
	return m_cprInfo.GetCPRLineVoxel();
}

bool DataManager::RotateCPR(float angle, PlaneType planeType)
{
	return m_cprInfo.RotateCPR(angle, planeType);
}

bool DataManager::GetCPRInfo(Point3d*& pPoints, Direction3d*& pDirs, int& nWidth, int& nHeight, PlaneType planeType)
{
	return m_cprInfo.GetCPRInfo(pPoints, pDirs, nWidth, nHeight, planeType);
}

int DataManager::GetDim( int index )
{
	return m_volInfo.GetDim(index);
}

double DataManager::GetSpacing(int index)
{
	return m_volInfo.GetSpacing(index);
}

double DataManager::GetMinSpacing()
{
	return m_volInfo.GetMinSpacing();
}

std::vector<Point3d> DataManager::GetVertexes()
{
	double xLen = GetDim(0) * GetSpacing(0);
	double yLen = GetDim(1) * GetSpacing(1);
	double zLen = GetDim(2) * GetSpacing(2);
	double spacing = GetMinSpacing();
	std::vector<Point3d> ptVertexes;
	ptVertexes.push_back(Point3d(0, 0, 0));
	ptVertexes.push_back(Point3d(xLen, 0, 0));
	ptVertexes.push_back(Point3d(0, yLen, 0));
	ptVertexes.push_back(Point3d(xLen, yLen, 0));
	ptVertexes.push_back(Point3d(0, 0, zLen));
	ptVertexes.push_back(Point3d(xLen, 0, zLen));
	ptVertexes.push_back(Point3d(0, yLen, zLen));
	ptVertexes.push_back(Point3d(xLen, yLen, zLen));
	return ptVertexes;
}

bool DataManager::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	if (planeType == PlaneVR)
	{
		nWidth = m_renderInfo.m_nWidth_VR;
		nHeight = m_renderInfo.m_nHeight_VR;
		return true;
	}
	else if (planeType == PlaneStretchedCPR || planeType == PlaneStraightenedCPR)
	{
		// return m_cprInfo.GetPlaneMaxSize(nWidth, nHeight, planeType);
	}
	else
	{
		PlaneInfo info;
		if (!GetPlaneInfo(planeType, info))
			return false;
		double xLen = GetDim(0) * GetSpacing(0);
		double yLen = GetDim(1) * GetSpacing(1);
		double zLen = GetDim(2) * GetSpacing(2);
		double spacing = GetMinSpacing();
		nWidth = int(sqrtf(xLen*xLen + yLen*yLen + zLen*zLen)/spacing) + 1;
		nHeight = nWidth;
		return true;
	}
	return false;
}

void DataManager::SetRenderType(RenderType type)
{
	m_renderType = type;
}

RenderType DataManager::GetRenderType()
{
	return m_renderType;
}

bool DataManager::SetVRWWWL(float fWW, float fWL)
{
	return SetVRWWWL(fWW, fWL, m_activeLabel);
}

bool DataManager::SetVRWWWL(float fWW, float fWL, unsigned char nLabel)
{
	if (nLabel < 0 || nLabel > MAXOBJECTCOUNT){
		return false;
	}
	if (fWW <= 0){
		fWW = 1.0;
	}
	m_objectInfos[nLabel].ww = fWW;
	m_objectInfos[nLabel].wl = fWL;

	UpdateAlphaWWWL();
	return true;
}

bool DataManager::SetObjectAlpha(float fAlpha)
{
	return SetObjectAlpha(fAlpha, m_activeLabel);
}

bool DataManager::SetObjectAlpha(float fAlpha, unsigned char nLabel)
{
	Logger::Info("DataManager::SetObjectAlpha: alpha[%.2f], label[%d]", fAlpha, nLabel);
	if (nLabel < 0 || nLabel > MAXOBJECTCOUNT){
		return false;
	}
	m_objectInfos[nLabel].alpha = fAlpha;

	UpdateAlphaWWWL();
	return true;
}

void DataManager::UpdateAlphaWWWL()
{
	for (std::map<unsigned char, ObjectInfo>::iterator iter = m_objectInfos.begin(); iter != m_objectInfos.end(); iter++)
	{
		unsigned char label = iter->first;
		ObjectInfo info = iter->second;
		m_AlphaAndWWWLInfo[label] = AlphaAndWWWL(info.alpha, info.ww, info.wl);
		Logger::Info("DataManager::UpdateAlphaWWWL: label[%d], alpha[%.2f], ww[%.2f], wl[%.2f]", label, info.alpha, info.ww, info.wl);
	}
}

bool DataManager::GetBatchDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, double fAngle, const PlaneType& planeType )
{
	Direction3d dirH_Plane, dirV_Plane;
	GetDirection3D(dirH_Plane, dirV_Plane, planeType);
	Direction3d dirA = dirH_Plane.cross(dirV_Plane);

	fAngle = fAngle/180*PI;
	double x = dirA.x();
	double y = dirA.y();
	double z = dirA.z();
	double xx = x*x;
	double yy = y*y;
	double zz = z*z;
	double sinV = sin(fAngle);
	double cosV = cos(fAngle);
	double cosVT = 1.0 - cosV;
	double m[3][3];
	m[0][0] = xx + (1-xx)*cosV;
	m[1][1] = yy + (1-yy)*cosV;
	m[2][2] = zz + (1-zz)*cosV;
	m[0][1] = x*y*cosVT - z*sinV;
	m[0][2] = x*z*cosVT + y*sinV;
	m[1][0] = x*y*cosVT + z*sinV;
	m[1][2] = y*z*cosVT - x*sinV;
	m[2][0] = x*z*cosVT - y*sinV;
	m[2][1] = y*z*cosVT + x*sinV;

	Point3d pt = Methods::GetTransferPoint(m, Point3d(dirH_Plane.x(), dirH_Plane.y(), dirH_Plane.z()));
	dir3dH = Direction3d(pt.x(), pt.y(), pt.z());
	dir3dV = dirA;
	return true;
}

// MPR
void DataManager::SetMPRType( MPRType type )
{
	m_mprInfo.SetMPRType(type);
}

bool DataManager::GetPlaneInfo(PlaneType planeType, PlaneInfo& info)
{
	return m_mprInfo.GetPlaneInfo(planeType, info);
}

void DataManager::UpdateThickness( double val )
{
	m_mprInfo.UpdateThickness(val);
}

void DataManager::SetThickness(double val, PlaneType planeType)
{
	m_mprInfo.SetThickness(val, planeType);
}

double DataManager::GetThickness(PlaneType planeType)
{
	return m_mprInfo.GetThickness(planeType);
}

double DataManager::GetPixelSpacing( PlaneType planeType )
{
	return m_mprInfo.GetPixelSpacing(planeType);
}

bool DataManager::GetPlaneSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	if (planeType >= PlaneAxial && planeType <= PlaneCoronalOblique)
	{
		return m_mprInfo.GetPlaneSize(nWidth, nHeight, planeType);
	}
	return false;
}

bool DataManager::GetPlaneNumber( int& nNumber, const PlaneType& planeType )
{
	return m_mprInfo.GetPlaneNumber(nNumber, planeType);
}

bool DataManager::GetPlaneIndex( int& index, const PlaneType& planeType )
{
	return m_mprInfo.GetPlaneIndex(index, planeType);
}

void DataManager::SetPlaneIndex( int index, PlaneType planeType )
{
	m_mprInfo.SetPlaneIndex(index, planeType);
}

bool DataManager::GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType )
{
	if (planeType == PlaneVR){
		memcpy(pMatirx, m_renderInfo.m_pRotateMatrix, 9 * sizeof(float));
		return true;
	}
	return m_mprInfo.GetPlaneRotateMatrix(pMatirx, planeType);
}

void DataManager::Browse( float fDelta, PlaneType planeType )
{
	m_mprInfo.Browse(fDelta, planeType);
}
void DataManager::PanCrossHair(float fx, float fy, PlaneType planeType)
{
	m_mprInfo.PanCrossHair(fx, fy, planeType);
}

void  DataManager::RotateCrossHair( float fAngle, PlaneType planeType )
{
	m_mprInfo.RotateCrossHair(fAngle, planeType);
}

bool DataManager::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	return m_mprInfo.GetCrossHairPoint(x, y, planeType);
}

bool DataManager::TransferImage2Voxel(double& x, double& y, double& z, double xImage, double yImage, PlaneType planeType)
{
	return m_mprInfo.TransferImage2Voxel(x, y, z, xImage, yImage, planeType);
}

bool DataManager::TransferImage2Voxel(Point3d& ptVoxel, double xImage, double yImage, PlaneType planeType)
{
	return m_mprInfo.TransferImage2Voxel(ptVoxel, xImage, yImage, planeType);
}

bool DataManager::GetDirection( Direction2d& dirH, Direction2d& dirV, const PlaneType& planeType )
{
	return m_mprInfo.GetDirection(dirH, dirV, planeType);
}

bool DataManager::GetDirection3D( Direction3d& dir3dH, Direction3d& dir3dV, const PlaneType& planeType )
{
	return m_mprInfo.GetDirection3D(dir3dH, dir3dV, planeType);
}

Point3d DataManager::GetCenterPointPlane(Direction3d dirN)
{
	return m_mprInfo.GetCenterPointPlane(dirN);
}

Point3d DataManager::GetCrossHair()
{
    return m_mprInfo.GetCrossHair();
}

void DataManager::SetCrossHair(Point3d pt)
{
    m_mprInfo.SetCrossHair(pt);
}

Point3d DataManager::GetCenterPoint()
{
    return m_mprInfo.GetCenterPoint();
}

void DataManager::ShowPlaneInVR(bool bShow)
{
	if (bShow)
	{
		Logger::Info("DataManager::ShowPlaneInVR");
		int nWidth = m_volInfo.GetDim(0);
		int nHeight = m_volInfo.GetDim(1);
		int nDepth = m_volInfo.GetDim(2);

		int xySize = nWidth*nHeight;
		int xzSize = nWidth*nDepth;
		int yzSize = nHeight*nDepth;

		std::shared_ptr<unsigned char> pData(new unsigned char[nWidth*nHeight*nDepth]);
		for (int z=200; z<203; z++)
		{
			for (int xy=0; xy<xySize; xy++){
				pData.get()[xy+z*xySize] = 1;
			}
		}

		for (int z=0; z<nDepth; z++){
			for (int y=200; y<203; y++){
				for (int x=0; x<nWidth; x++){
					pData.get()[z*xySize+y*nWidth+x] = 1;
				}
			}
		}
		for (int z=0; z<nDepth; z++){
			for (int y=0; y<nHeight; y++){
				for (int x=250; x<253; x++){
					pData.get()[z*xySize+y*nWidth+x] = 1;
				}
			}
		}

		m_planeLabel = AddNewObjectMask(pData, nWidth, nHeight, nDepth);
		SetVRWWWL(1000, -40);
		std::map<int, RGBA> ctrlPts;
		ctrlPts[0] = RGBA(0, 0, 0, 1);
		ctrlPts[99] = RGBA(1, 1, 1, 1);
		SetObjectAlpha(0.6, m_planeLabel);
	}
}

bool DataManager::AddAnnotation(PlaneType planeType, std::string txt, int x, int y, FontSize fontSize, AnnotationFormat annoFormat, RGBA clr)
{
	return GetAnnotationInfo().AddAnnotation(PlaneVR, txt, x, y, fontSize, annoFormat, clr);
}

bool DataManager::RemovePlaneAnnotations(PlaneType planeType)
{
	return GetAnnotationInfo().RemovePlaneAnnotations(planeType);
}

bool DataManager::RemoveAllAnnotations()
{
	return GetAnnotationInfo().RemoveAllAnnotations();
}

bool DataManager::EnableLayer(LayerType layerType, bool bEnable)
{
	m_layerEnable[layerType] = bEnable;
	return true;
}

bool DataManager::IsLayerEnable(LayerType layerType)
{
	if (m_layerEnable.find(layerType) != m_layerEnable.end()){
		return m_layerEnable[layerType];
	}
	return true;
}

void DataManager::SetCPRLineColor(RGBA clr)
{
	m_cprInfo.SetLineColor(clr);
}