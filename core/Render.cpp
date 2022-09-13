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

#include "Render.h"
#include <driver_types.h>
#include "vector_types.h"
#include "StopWatch.h"
#include "TransferFunction.h"
#include "Logger.h"

using namespace MonkeyGL;

extern "C"
void cu_render(
	unsigned char* pVR, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	cudaTextureObject_t maskTexture,
	cudaTextureObjects transferFuncTextures,
	AlphaAndWWWLInfo alphaAndWWWLInfo, 
	float3 f3maxLenSpacing,
	float3 f3Spacing,
	float3 f3SpacingVoxel,
	float xTranslate, 
	float yTranslate, 
	float scale, 
	float3x3 transformMatrix, 
	bool invertZ,
	VOI m_voi, 
	RGBA colorBG,
	RenderType type
);

extern "C"
void cu_renderPlane_MIP(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing,
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
	);
extern "C"
void cu_renderPlane_MinIP(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing,
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
);
extern "C"
void cu_renderPlane_Average(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing,
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
);

extern "C"
void cu_renderCPR(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	double* pPoints, 
	double* pDirs, 
	bool invertZ
);

extern "C" 
void cu_test_3d( short* h_volumeData, cudaExtent volumeSize );
extern "C"
void cu_init_id();
extern "C" 
bool cu_set_1d( float* h_volumeData, int nLen, unsigned char nLabel );
extern "C" 
void cu_test_1d( int nLen,  unsigned char nLabel );

Render::Render(void)
{
	m_renderType = RenderTypeVR;
	Init();

	// testcuda();
}


void Render::testcuda()
{
#if 1
	int nWidth = 512;
	int nHeight = 512;
	int nDepth = 200;
	short* pData = new short[nWidth*nHeight*nDepth];
	for(int i=0; i<nDepth; i++)
	{
		for(int j=0; j<nWidth*nHeight; j++){
			pData[i*nHeight*nWidth + j] = j+i;
		}
	}
	m_VolumeSize.width = nWidth;
	m_VolumeSize.height = nHeight;
	m_VolumeSize.depth = nDepth;

	cu_test_3d(pData, m_VolumeSize);

	delete [] pData;
#else
	cu_init_id();
	int nLen = 100;
	float* pData = new float[nLen*4];
	for(int i=0; i<nLen; i++){
		pData[4*i] = i;
		pData[4*i+1] = i;
		pData[4*i+2] = i;
		pData[4*i+3] = i+1;
	}
	cu_set_1d(pData, nLen, 1);

	for(int i=0; i<nLen; i++){
		pData[4*i] = i+20;
		pData[4*i+1] = i+20;
		pData[4*i+2] = i+20;
		pData[4*i+3] = i+20+1;
	}
	cu_set_1d(pData, nLen, 10);

	for (int l=0; l<10000; l++)
	{
		for (int i=0; i<12; i++)
			cu_test_1d(nLen, i);
	}

	delete [] pData;
#endif
}


Render::~Render(void)
{
}

void Render::Init()
{
	m_fTotalXTranslate = 0.0f;
	m_fTotalYTranslate = 0.0f;
	m_fTotalScale = 1.0f;

	Methods::SetSeg(m_pRotateMatrix, 3);
	Methods::SetSeg(m_pTransposRotateMatrix, 3);
	Methods::SetSeg(m_pTransformMatrix, 3);
	Methods::SetSeg(m_pTransposeTransformMatrix, 3);
	
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

void Render::Reset()
{
	IRender::Reset();
	Init();
}

bool Render::SetTransferFunc( std::map<int, RGBA> ctrlPoints )
{
	if (!IRender::SetTransferFunc(ctrlPoints)){
		return false;
	}
	UpdateTransferFunctions();
	return true;
}

bool Render::SetTransferFunc(std::map<int, RGBA> ctrlPoints, unsigned char nLabel )
{
	if (!IRender::SetTransferFunc(ctrlPoints, nLabel)){
		return false;
	}
	UpdateTransferFunctions();
	return true;
}

bool Render::SetTransferFunc(std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints)
{
	if (!IRender::SetTransferFunc(rgbPoints, alphaPoints)){
		return false;
	}
	UpdateTransferFunctions();
	return true;
}

bool Render::SetTransferFunc(std::map<int, RGBA> rgbPoints, std::map<int, float> alphaPoints, unsigned char nLabel)
{
	if (!IRender::SetTransferFunc(rgbPoints, alphaPoints, nLabel)){
		return false;
	}
	UpdateTransferFunctions();
	return true;
}

bool Render::LoadTransferFunction(const char* szFile)
{
	if (!IRender::LoadTransferFunction(szFile)){
		return false;
	}
	UpdateTransferFunctions();
	return true;
}

void Render::UpdateTransferFunctions()
{
	std::shared_ptr<RGBA> ptfBuffer(NULL);
	int ntfLength = 0;

	std::map<unsigned char, ObjectInfo> objectInfos = m_dataMan.GetObjectInfos();
	for (std::map<unsigned char, ObjectInfo>::iterator iter=objectInfos.begin(); iter!=objectInfos.end(); iter++){
		if (iter->second.GetTransferFunction(ptfBuffer, ntfLength))
		{
			m_cuDataInfo.SetTransferFunction((float*)(ptfBuffer.get()), ntfLength, iter->first);
		}
	}
}

bool Render::SetVRWWWL(float fWW, float fWL)
{
	if (!IRender::SetVRWWWL(fWW, fWL)){
		return false;
	}
	UpdateAlphaWWWL();
	return true;
}

bool Render::SetVRWWWL(float fWW, float fWL, unsigned char nLabel)
{
	if (!IRender::SetVRWWWL(fWW, fWL, nLabel)){
		return false;
	}
	UpdateAlphaWWWL();
	return true;
}

bool Render::SetObjectAlpha(float fAlpha)
{
	if (!IRender::SetObjectAlpha(fAlpha)){
		return false;
	}
	UpdateAlphaWWWL();
	return true;
}

bool Render::SetObjectAlpha(float fAlpha, unsigned char nLabel)
{
	if (!IRender::SetObjectAlpha(fAlpha, nLabel)){
		return false;
	}
	UpdateAlphaWWWL();
	return true;
}

void Render::UpdateAlphaWWWL()
{
	std::map<unsigned char, ObjectInfo> objectInfos = m_dataMan.GetObjectInfos();
	for (std::map<unsigned char, ObjectInfo>::iterator iter=objectInfos.begin(); iter!=objectInfos.end(); iter++){
		unsigned char label = iter->first;
		ObjectInfo info = iter->second;
		m_AlphaAndWWWLInfo[label] = AlphaAndWWWL(info.alpha, info.ww, info.wl);
		Logger::Info("Render::UpdateAlphaWWWL: label[%d], alpha[%.2f], ww[%.2f], wl[%.2f]", label, info.alpha, info.ww, info.wl);
	}
}

bool Render::SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth)
{
	if (!IRender::SetVolumeData(pData, nWidth, nHeight, nDepth))
		return false;

	m_VolumeSize.width = m_dataMan.GetDim(0);
	m_VolumeSize.height = m_dataMan.GetDim(1);
	m_VolumeSize.depth = m_dataMan.GetDim(2);

	m_cuDataInfo.CopyVolumeData(m_dataMan.GetVolumeData().get(), m_VolumeSize);

	return true;
}

unsigned char Render::AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth)
{
	unsigned char nLabel = IRender::AddNewObjectMask(pData, nWidth, nHeight, nDepth);
	if (nLabel == 0)
		return 0;

	m_cuDataInfo.CopyMaskData(m_dataMan.GetMaskData().get(), m_VolumeSize);

	return nLabel;
}

unsigned char Render::AddObjectMaskFile(const char* szFile)
{
	unsigned char nLabel = IRender::AddObjectMaskFile(szFile);
	if (nLabel == 0)
		return 0;

	m_cuDataInfo.CopyMaskData(m_dataMan.GetMaskData().get(), m_VolumeSize);

	return nLabel;
}

bool Render::UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel)
{
	if (!IRender::AddNewObjectMask(pData, nWidth, nHeight, nDepth))
		return false;

	m_cuDataInfo.CopyMaskData(m_dataMan.GetMaskData().get(), m_VolumeSize);

	return true;
}

void Render::LoadVolumeFile( const char* szFile )
{
	Logger::Info("load volume file: %s", szFile);
	
	IRender::LoadVolumeFile(szFile);

	m_VolumeSize.width = m_dataMan.GetDim(0);
	m_VolumeSize.height = m_dataMan.GetDim(1);
	m_VolumeSize.depth = m_dataMan.GetDim(2);

	m_cuDataInfo.CopyVolumeData(m_dataMan.GetVolumeData().get(), m_VolumeSize);
	InitCommon(m_dataMan.GetSpacing(0), m_dataMan.GetSpacing(1), m_dataMan.GetSpacing(2), m_VolumeSize);
}

void Render::NormalizeVOI()
{
	m_voi_Normalize.left = (int)m_fVOI_xStart;
	m_voi_Normalize.right = (int)m_fVOI_xEnd;
	m_voi_Normalize.posterior = (int)m_fVOI_yStart;
	m_voi_Normalize.anterior = (int)m_fVOI_yEnd;
	m_voi_Normalize.head = (int)m_fVOI_zStart;
	m_voi_Normalize.foot = (int)m_fVOI_zEnd;
}

void Render::SetSpacing( double x, double y, double z )
{
	IRender::SetSpacing(x, y, z);
	InitCommon(x, y, z, m_VolumeSize);
}

bool Render::GetPlaneData( std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	if (
		PlaneAxial == planeType ||
		PlaneAxialOblique == planeType ||
		PlaneSagittal == planeType ||
		PlaneSagittalOblique == planeType ||
		PlaneCoronal == planeType ||
		PlaneCoronalOblique == planeType
	)
	{
		return GetMPRPlaneData(pData, nWidth, nHeight, planeType);
	}
	else if (PlaneStretchedCPR == planeType || PlaneStraightenedCPR == planeType) 
	{
		return GetCPRPlaneData(pData, nWidth, nHeight, planeType);
	}

	return false;
}

bool Render::GetMPRPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	if (!m_dataMan.GetPlaneSize(nWidth, nHeight, planeType))
		return false;

	if (nWidth % 2) // just for fpng
	{
		nWidth += 1;
	}
	pData.reset(new short[nWidth*nHeight]);

	PlaneInfo info;
	if (!m_dataMan.GetPlaneInfo(planeType, info))
		return false;

	Direction3d& dirH = info.m_dirH;
	Direction3d& dirV = info.m_dirV;
	Direction3d dirN = info.GetNormDirection();
	double fPixelSpacing = info.m_fPixelSpacing;
	Point3d ptCenter = m_dataMan.GetCenterPointPlane(dirN);
	Point3d ptLeftTop = ptCenter - dirH*(0.5*nWidth*fPixelSpacing);
	ptLeftTop = ptLeftTop - dirV*(0.5*nHeight*fPixelSpacing);

	double fSliceThickness = info.m_fSliceThickness;
	int nSliceNum = fSliceThickness/fPixelSpacing;
	nSliceNum = nSliceNum<1 ? 1:nSliceNum;
	float halfNum = 1.0f*(nSliceNum-1)/2;

	float3 dirH_cu;
	dirH_cu.x = info.m_dirH.x();
	dirH_cu.y = info.m_dirH.y();
	dirH_cu.z = info.m_dirH.z();
	float3 dirV_cu;
	dirV_cu.x = info.m_dirV.x();
	dirV_cu.y = info.m_dirV.y();
	dirV_cu.z = info.m_dirV.z();
	float3 dirN_cu;
	dirN_cu.x = dirN.x();
	dirN_cu.y = dirN.y();
	dirN_cu.z = dirN.z();

	float3 ptLeftTop_cu;
	ptLeftTop_cu.x = ptLeftTop[0];
	ptLeftTop_cu.y = ptLeftTop[1];
	ptLeftTop_cu.z = ptLeftTop[2];

	switch (info.m_MPRType)
	{
	case MPRTypeAverage:
		{
			cu_renderPlane_Average(
				pData.get(), 
				nWidth, 
				nHeight, 
				m_cuDataInfo.m_h_volumeTexture, 
				m_VolumeSize, 
				m_f3Spacing,
				dirH_cu, 
				dirV_cu, 
				dirN_cu, 
				ptLeftTop_cu, 
				info.m_fPixelSpacing, 
				m_dataMan.Need2InvertZ(), 
				halfNum
			);
			return true;
		}
		break;
	case MPRTypeMIP:
		{
			cu_renderPlane_MIP(
				pData.get(), 
				nWidth, 
				nHeight, 
				m_cuDataInfo.m_h_volumeTexture, 
				m_VolumeSize, 
				m_f3Spacing,
				dirH_cu, 
				dirV_cu, 
				dirN_cu, 
				ptLeftTop_cu, 
				info.m_fPixelSpacing, 
				m_dataMan.Need2InvertZ(), 
				halfNum
			);
			return true;
		}
		break;
	case MPRTypeMinIP:
		{
			cu_renderPlane_MinIP(
				pData.get(), 
				nWidth, 
				nHeight, 
				m_cuDataInfo.m_h_volumeTexture, 
				m_VolumeSize, 
				m_f3Spacing,
				dirH_cu, 
				dirV_cu, 
				dirN_cu, 
				ptLeftTop_cu, 
				info.m_fPixelSpacing, 
				m_dataMan.Need2InvertZ(), 
				halfNum
			);
			return true;
		}
		break;
	default:
		break;
	}
	return false;
}
bool Render::GetCPRPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType)
{
	StopWatch sw("Render::GetCPRPlaneData: PlaneType[%s]", PlaneTypeName(planeType).c_str());

	Point3d* pPoints = NULL;
	Direction3d* pDirs = NULL;

	if (!m_dataMan.GetCPRInfo(pPoints, pDirs, nWidth, nHeight, planeType))
		return false;

	pData.reset(new short[nWidth*nHeight]);

	cu_renderCPR(
		pData.get(), 
		nWidth, 
		nHeight, 
		m_cuDataInfo.m_h_volumeTexture,
		m_VolumeSize,
		(double*)pPoints, 
		(double*)pDirs, 
		m_dataMan.Need2InvertZ()
	);

	delete [] pPoints;
	delete [] pDirs;

	return true;
}

bool Render::GetPlaneMaxSize( int& nWidth, int& nHeight, const PlaneType& planeType )
{
	return IRender::GetPlaneMaxSize(nWidth, nHeight, planeType);
}

bool Render::GetCrossHairPoint( double& x, double& y, const PlaneType& planeType )
{
	if (PlaneVR == planeType)
	{
		int nWidth = 0;
		int nHeight = 0;
		m_dataMan.GetPlaneSize(nWidth, nHeight, planeType);
		Point3d ptCrossHair = m_dataMan.GetCrossHair();
		Point3d ptDelta = ptCrossHair - m_dataMan.GetCenterPoint();
		Point3d ptRotate = Methods::matrixMul(m_pTransposeTransformMatrix, ptDelta);

		double x = ptRotate.x();
		double z = ptRotate.z();

		double xLen = m_dataMan.GetDim(0)*m_dataMan.GetSpacing(0);
		double zLen = m_dataMan.GetDim(2)*m_dataMan.GetSpacing(2);

		double spacing = (xLen/nWidth)>(zLen/nHeight) ? (xLen/nWidth) : (zLen/nHeight);
		
		x = (nWidth-1)/2.0 + (x/spacing) + m_fTotalXTranslate;
		y = (nHeight-1)/2.0 + (z/spacing) + m_fTotalYTranslate;
	}
	else
	{
		return m_dataMan.GetCrossHairPoint(x, y, planeType);
	}
	return true;
}

void Render::PanCrossHair( float fx, float fy, PlaneType planeType )
{
	if (PlaneVR == planeType)
	{
	}
	else
	{
		m_dataMan.PanCrossHair(fx, fy, planeType);
	}
}

bool Render::GetVRData( unsigned char* pVR, int nWidth, int nHeight )
{
	m_fVOI_xStart = 0;
	m_fVOI_xEnd = m_VolumeSize.width - 1;
	m_fVOI_yStart = 0;
	m_fVOI_yEnd = m_VolumeSize.height - 1;
	m_fVOI_zStart = 0;
	m_fVOI_zEnd = m_VolumeSize.depth - 1;
	NormalizeVOI();

	cu_render(
		pVR, 
		nWidth, 
		nHeight, 
		m_cuDataInfo.m_h_volumeTexture,
		m_VolumeSize,
		m_cuDataInfo.m_h_maskTexture,
		m_cuDataInfo.m_h_transferFuncTextures,
		m_AlphaAndWWWLInfo,
		m_f3maxLenSpacing,
		m_f3Spacing,
		m_f3SpacingVoxel,
		m_fTotalXTranslate, 
		m_fTotalYTranslate, 
		m_fTotalScale,
		m_cuDataInfo.m_h_transformMatrix, 
		m_dataMan.Need2InvertZ(),
		m_voi_Normalize, 
		m_dataMan.GetColorBackground(), 
		m_renderType
	);

	MergeOrientationBox(pVR, nWidth, nHeight);

	return true;
}

void Render::MergeOrientationBox(unsigned char* pVR, int nWidth, int nHeight)
{
	int nW_Box = 64, nH_Box=64;
	if (nWidth <= nW_Box || nHeight <= nH_Box){
		return;
	}

	{
		int w = 512;
		int h = 512;
		std::vector<Facet2D> facet2Ds = GetMeshPoints(w, h);
		std::shared_ptr<float> pImage(new float[w * h]);
		memset(pImage.get(), 0, w * h*sizeof(float));
		std::shared_ptr<float> pZBuffer(new float[w * h]);
		memset(pZBuffer.get(), 0, w * h*sizeof(float));

		for (size_t i=0; i<facet2Ds.size(); i++){					
			Facet2D& facet2D = facet2Ds[i];

			Point2f& v1 = facet2D.v1;
			Point2f& v2 = facet2D.v2;
			Point2f& v3 = facet2D.v3;
			float& zBuffer = facet2D.zBuffer;
			Methods::FillHoleInImage_Ch1(pImage.get(), pZBuffer.get(), w, h, facet2D.diffuse, zBuffer, v1, v2, v3);
		}

		RGB clr(0.902, 0.902, 0.302);
		for (int y=nHeight-nH_Box; y<nHeight; y++){
			int yIdx = (y + nH_Box - nHeight) * 8;
			for (int x=nWidth-nW_Box; x<nWidth; x++){
				int xIdx = (x + nW_Box - nWidth) * 8;
				float diffuse = pImage.get()[yIdx*w+xIdx];
				if (diffuse > 0){
					RGB clrTemp = clr * diffuse;	
					int red = int(clrTemp.red * 255);
					int green = int(clrTemp.green * 255);
					int blue = int(clrTemp.blue * 255);
					pVR[3*(y*nWidth+x)] = red;
					pVR[3*(y*nWidth+x)+1] = green;
					pVR[3*(y*nWidth+x)+2] = blue;
				}
			}
		}
	}
}

bool Render::GetBatchData( std::vector<short*>& vecBatchData, BatchInfo batchInfo )
{
	for (int i=0; i<vecBatchData.size(); i++)
	{
		if (nullptr != vecBatchData[i])
		{
			delete [] vecBatchData[i];
			vecBatchData[i] = nullptr;
		}
	}
	vecBatchData.clear();

	int nWidth = batchInfo.Width();
	int nHeight = batchInfo.Height();

	Direction3d& dirH = batchInfo.m_dirH;
	Direction3d& dirV = batchInfo.m_dirV;
	Direction3d dirN = dirH.cross(dirV);
	int& nNum = batchInfo.m_nNum;
	Point3d& ptCenter = batchInfo.m_ptCenter;
	double& fSliceDist = batchInfo.m_fSliceDistance;

	double fSliceThickness = batchInfo.m_fSliceThickness;
	int nSliceNum = fSliceThickness/batchInfo.m_fPixelSpacing;
	nSliceNum = nSliceNum<1 ? 1:nSliceNum;
	float halfNum = 1.0f*(nSliceNum-1)/2;

	float3 dirH_cu;
	dirH_cu.x = dirH.x();
	dirH_cu.y = dirH.y();
	dirH_cu.z = dirH.z();
	float3 dirV_cu;
	dirV_cu.x = dirV.x();
	dirV_cu.y = dirV.y();
	dirV_cu.z = dirV.z();
	float3 dirN_cu;
	dirN_cu.x = dirN.x();
	dirN_cu.y = dirN.y();
	dirN_cu.z = dirN.z();

	for (int i=-nNum/2; i<=nNum/2; i++)
	{
		Point3d ptCenterSlice = ptCenter;
		double fDist = 0;
		if (nNum % 2 == 0)
		{
			if (i == 0)
				continue;
			if (i < 0)
			{
				fDist = (i+0.5) * fSliceDist;
			}
			else
			{
				fDist = (i-0.5) * fSliceDist;
			}
		}
		else
		{
			fDist = i * fSliceDist;
		}

		ptCenterSlice = ptCenterSlice + dirN * fDist;

		Point3d ptLeftTop = ptCenterSlice - dirH*(0.5*batchInfo.m_fLengthH);
		ptLeftTop = ptLeftTop - dirV*(0.5*batchInfo.m_fLengthV);

		float3 ptLeftTop_cu;
		ptLeftTop_cu.x = ptLeftTop[0];
		ptLeftTop_cu.y = ptLeftTop[1];
		ptLeftTop_cu.z = ptLeftTop[2];

		short* pData = new short[nWidth*nHeight];
		switch (batchInfo.m_MPRType)
		{
		case MPRTypeAverage:
			{
				cu_renderPlane_Average(
					pData, 
					nWidth, 
					nHeight, 
					m_cuDataInfo.m_h_volumeTexture, 
					m_VolumeSize, 
					m_f3Spacing, 
					dirH_cu, 
					dirV_cu, 
					dirN_cu, 
					ptLeftTop_cu, 
					batchInfo.m_fPixelSpacing, 
					m_dataMan.Need2InvertZ(), 
					halfNum
				);
			}
			break;
		default:
			return false;
		}
		vecBatchData.push_back(pData);
	}
	return true;
}

bool Render::GetPlaneRotateMatrix( float* pMatirx, PlaneType planeType )
{
	if (planeType == PlaneVR)
	{
		memcpy(pMatirx, m_pRotateMatrix, 9*sizeof(float));
		return true;
	}
	return IRender::GetPlaneRotateMatrix(pMatirx, planeType);
}

void Render::Anterior()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 0.0f, m_fTotalScale);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

void Render::Posterior()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 180.0f, m_fTotalScale);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

void Render::Left()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, -90.0f, m_fTotalScale);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

void Render::Right()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 90.0f, m_fTotalScale);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

void Render::Head()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 90.0f, 180.0f, m_fTotalScale);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

void Render::Foot()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, -90.0f, 0.0f, m_fTotalScale);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

void Render::Rotate( float fxRotate, float fyRotate )
{
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, fyRotate, fxRotate, 1.0f);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
}

float Render::Zoom( float ratio)
{
	m_fTotalScale *= ratio;
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 0.0f, ratio);
	m_cuDataInfo.CopyOperatorMatrix(m_pTransformMatrix, m_pTransposeTransformMatrix);
	return m_fTotalScale;
}

float Render::GetZoomRatio()
{
	return m_fTotalScale;
}

void Render::Pan(float fxShift, float fyShift)
{
	m_fTotalXTranslate += fxShift;
	m_fTotalYTranslate += fyShift;
}

bool Render::TransferVoxel2ImageInVR(float& fx, float& fy, int nWidth, int nHeight, Point3d ptVoxel)
{
	double fxSpacing = m_dataMan.GetSpacing(0);
	double fySpacing = m_dataMan.GetSpacing(1);
	double fzSpacing = m_dataMan.GetSpacing(2);

	float fMaxLen = max(m_VolumeSize.width*fxSpacing, max(m_VolumeSize.height*fySpacing, m_VolumeSize.depth*fzSpacing));
	Point3d maxper(fMaxLen/(m_VolumeSize.width*fxSpacing), fMaxLen/(m_VolumeSize.height*fySpacing), fMaxLen/(m_VolumeSize.depth*fzSpacing));

	Point3d pt(ptVoxel.x()/m_VolumeSize.width, ptVoxel.y()/m_VolumeSize.height, ptVoxel.z()/m_VolumeSize.depth);
	if (m_dataMan.Need2InvertZ()){
		pt.SetZ(1.0 - pt.z());
	}
	pt -= Point3d(0.5, 0.5, 0.5);
	pt /= maxper;

	pt = Methods::matrixMul(m_pTransposeTransformMatrix, pt);
	fx = (pt.x() + 0.5) * nWidth + m_fTotalXTranslate;
	fy = (pt.z() + 0.5) * nHeight + m_fTotalYTranslate;

	return true;
}

void Render::ShowPlaneInVR(bool bShow)
{
	IRender::ShowPlaneInVR(bShow);
	m_cuDataInfo.CopyMaskData(m_dataMan.GetMaskData().get(), m_VolumeSize);
	UpdateAlphaWWWL();
	UpdateTransferFunctions();
}



Point2f TransferPoint2D(Point2f pt, int nWidth, int nHeight){
	float r = nHeight;
	Point2f ptOut = pt;
	ptOut *= r;
	ptOut += Point2f(nWidth/2.0, nHeight/2.0);
	return ptOut;
}

std::vector<Facet2D> Render::GetMeshPoints(int nWidth, int nHeight)
{
	std::vector<Facet3D> facet3Ds = m_marchingCube.GetMesh();
	std::vector<Facet2D> facet2Ds;
    for (size_t i=0; i<facet3Ds.size(); i++){
		Facet3D& facet3D = facet3Ds[i];
        
		Point3f v1 = Methods::GetTransferPointf(m_pRotateMatrix, facet3D.v1);
		Point3f v2 = Methods::GetTransferPointf(m_pRotateMatrix, facet3D.v2); 
		Point3f v3 = Methods::GetTransferPointf(m_pRotateMatrix, facet3D.v3); 

		Direction3f n = Direction3f(v3, v1).cross(Direction3f(v3, v2));
		float d = n.dot(Direction3f(0, -1, 0));
        if (d < 0){
            continue;
        }

        Facet2D facet2D;
        facet2D.diffuse = d;
		facet2D.zBuffer = (v1.y() + v2.y() + v3.y()) / 3 + 10000;
        facet2D.v1 = TransferPoint2D(Point2f(v1.x(), v1.z()), nWidth, nHeight);
        facet2D.v2 = TransferPoint2D(Point2f(v2.x(), v2.z()), nWidth, nHeight);
        facet2D.v3 = TransferPoint2D(Point2f(v3.x(), v3.z()), nWidth, nHeight);
        facet2Ds.push_back(facet2D);
    }
    return facet2Ds;
}