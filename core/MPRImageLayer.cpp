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

#include "MPRImageLayer.h"
#include <driver_types.h>
#include "vector_types.h"
#include "DataManager.h"

using namespace MonkeyGL;

extern "C" void cu_renderPlane_MIP(
	short *pData,
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
	float halfNum);

extern "C" void cu_renderPlane_MinIP(
	short *pData,
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
	float halfNum);

extern "C" void cu_renderPlane_Average(
	short *pData,
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
	float halfNum);

MPRImageLayer::MPRImageLayer()
: ImageLayer()
{
}

MPRImageLayer::~MPRImageLayer()
{
}

bool MPRImageLayer::GetGrayscaleData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, PlaneType planeType)
{
	if (!DataManager::Instance()->IsLayerEnable(GetLayerType())){
		return false;
	}
	
    if (planeType < PlaneAxial || planeType > PlaneCoronalOblique){
        return false;
    }

	if (!DataManager::Instance()->GetPlaneSize(nWidth, nHeight, planeType)){
		return false;
	}

	if (nWidth % 2) // just for fpng
	{
		nWidth += 1;
	}

    pData.reset(new short[nWidth * nHeight]);

	PlaneInfo info;
	if (!DataManager::Instance()->GetPlaneInfo(planeType, info))
		return false;

	Direction3d &dirH = info.m_dirH;
	Direction3d &dirV = info.m_dirV;
	Direction3d dirN = info.GetNormDirection();
	double fPixelSpacing = info.m_fPixelSpacing;
	Point3d ptCenter = DataManager::Instance()->GetCenterPointPlane(dirN);
	Point3d ptLeftTop = ptCenter - dirH * (0.5 * nWidth * fPixelSpacing);
	ptLeftTop = ptLeftTop - dirV * (0.5 * nHeight * fPixelSpacing);

	double fSliceThickness = info.m_fSliceThickness;
	int nSliceNum = fSliceThickness / fPixelSpacing;
	nSliceNum = nSliceNum < 1 ? 1 : nSliceNum;
	float halfNum = 1.0f * (nSliceNum - 1) / 2;

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
			DataManager::Instance()->GetCuDataManager().GetVolumeTexture(),
			DataManager::Instance()->GetVolumeSize(),
			DataManager::Instance()->GetSpacing(),
			dirH_cu,
			dirV_cu,
			dirN_cu,
			ptLeftTop_cu,
			info.m_fPixelSpacing,
			DataManager::Instance()->Need2InvertZ(),
			halfNum);
		return true;
	}
	case MPRTypeMIP:
	{
		cu_renderPlane_MIP(
			pData.get(),
			nWidth,
			nHeight,
			DataManager::Instance()->GetCuDataManager().GetVolumeTexture(),
			DataManager::Instance()->GetVolumeSize(),
			DataManager::Instance()->GetSpacing(),
			dirH_cu,
			dirV_cu,
			dirN_cu,
			ptLeftTop_cu,
			info.m_fPixelSpacing,
			DataManager::Instance()->Need2InvertZ(),
			halfNum);
		return true;
	}
	case MPRTypeMinIP:
	{
		cu_renderPlane_MinIP(
			pData.get(),
			nWidth,
			nHeight,
			DataManager::Instance()->GetCuDataManager().GetVolumeTexture(),
			DataManager::Instance()->GetVolumeSize(),
			DataManager::Instance()->GetSpacing(),
			dirH_cu,
			dirV_cu,
			dirN_cu,
			ptLeftTop_cu,
			info.m_fPixelSpacing,
			DataManager::Instance()->Need2InvertZ(),
			halfNum);
		return true;
	}
	default:
		break;
	}

    return false;
}