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
#include "StopWatch.h"
#include "TransferFunction.h"
#include "Logger.h"
#include "DataManager.h"

using namespace MonkeyGL;

Render::Render(void)
{
}

Render::~Render(void)
{
}

bool Render::GetPlaneData(std::shared_ptr<short> &pData, int &nWidth, int &nHeight, const PlaneType &planeType)
{
	if (
		PlaneAxial == planeType ||
		PlaneAxialOblique == planeType ||
		PlaneSagittal == planeType ||
		PlaneSagittalOblique == planeType ||
		PlaneCoronal == planeType ||
		PlaneCoronalOblique == planeType)
	{
		return GetMPRPlaneData(pData, nWidth, nHeight, planeType);
	}
	else if (PlaneStretchedCPR == planeType || PlaneStraightenedCPR == planeType)
	{
		return GetCPRPlaneData(pData, nWidth, nHeight, planeType);
	}

	return false;
}

bool Render::GetMPRPlaneData(std::shared_ptr<short> &pData, int &nWidth, int &nHeight, const PlaneType &planeType)
{
	StopWatch sw("Render::GetMPRPlaneData: PlaneType[%s]", PlaneTypeName(planeType).c_str());
	return m_mprProvider.GetGrayscaleData(pData, nWidth, nHeight, planeType);
}

bool Render::GetCPRPlaneData(std::shared_ptr<short> &pData, int &nWidth, int &nHeight, const PlaneType &planeType)
{
	StopWatch sw("Render::GetCPRPlaneData: PlaneType[%s]", PlaneTypeName(planeType).c_str());
	return m_cprProvider.GetGrayscaleData(pData, nWidth, nHeight, planeType);
}

bool Render::GetCrossHairPoint(double &x, double &y, const PlaneType &planeType)
{
	if (PlaneVR == planeType)
	{
		// int nWidth = 0;
		// int nHeight = 0;
		// DataManager::Instance()->GetPlaneSize(nWidth, nHeight, planeType);
		// Point3d ptCrossHair = DataManager::Instance()->GetCrossHair();
		// Point3d ptDelta = ptCrossHair - DataManager::Instance()->GetCenterPoint();
		// Point3d ptRotate = Methods::MatrixMul(m_pTransposeTransformMatrix, ptDelta);

		// double x = ptRotate.x();
		// double z = ptRotate.z();

		// double xLen = DataManager::Instance()->GetDim(0) * DataManager::Instance()->GetSpacing(0);
		// double zLen = DataManager::Instance()->GetDim(2) * DataManager::Instance()->GetSpacing(2);

		// double spacing = (xLen / nWidth) > (zLen / nHeight) ? (xLen / nWidth) : (zLen / nHeight);

		// x = (nWidth - 1) / 2.0 + (x / spacing) + m_fTotalXTranslate;
		// y = (nHeight - 1) / 2.0 + (z / spacing) + m_fTotalYTranslate;
	}
	else
	{
		return DataManager::Instance()->GetCrossHairPoint(x, y, planeType);
	}
	return true;
}

void Render::PanCrossHair(float fx, float fy, PlaneType planeType)
{
	if (PlaneVR == planeType)
	{
	}
	else
	{
		DataManager::Instance()->PanCrossHair(fx, fy, planeType);
	}
}

bool Render::GetVRData(std::shared_ptr<unsigned char>& pData, int nWidth, int nHeight)
{
	m_vrProvider.GetRGBData(pData, nWidth, nHeight, PlaneVR);
	return true;
}

bool Render::GetBatchData(std::vector<short *> &vecBatchData, BatchInfo batchInfo)
{
	// for (int i = 0; i < vecBatchData.size(); i++)
	// {
	// 	if (nullptr != vecBatchData[i])
	// 	{
	// 		delete[] vecBatchData[i];
	// 		vecBatchData[i] = nullptr;
	// 	}
	// }
	// vecBatchData.clear();

	// int nWidth = batchInfo.Width();
	// int nHeight = batchInfo.Height();

	// Direction3d &dirH = batchInfo.m_dirH;
	// Direction3d &dirV = batchInfo.m_dirV;
	// Direction3d dirN = dirH.cross(dirV);
	// int &nNum = batchInfo.m_nNum;
	// Point3d &ptCenter = batchInfo.m_ptCenter;
	// double &fSliceDist = batchInfo.m_fSliceDistance;

	// double fSliceThickness = batchInfo.m_fSliceThickness;
	// int nSliceNum = fSliceThickness / batchInfo.m_fPixelSpacing;
	// nSliceNum = nSliceNum < 1 ? 1 : nSliceNum;
	// float halfNum = 1.0f * (nSliceNum - 1) / 2;

	// float3 dirH_cu;
	// dirH_cu.x = dirH.x();
	// dirH_cu.y = dirH.y();
	// dirH_cu.z = dirH.z();
	// float3 dirV_cu;
	// dirV_cu.x = dirV.x();
	// dirV_cu.y = dirV.y();
	// dirV_cu.z = dirV.z();
	// float3 dirN_cu;
	// dirN_cu.x = dirN.x();
	// dirN_cu.y = dirN.y();
	// dirN_cu.z = dirN.z();

	// for (int i = -nNum / 2; i <= nNum / 2; i++)
	// {
	// 	Point3d ptCenterSlice = ptCenter;
	// 	double fDist = 0;
	// 	if (nNum % 2 == 0)
	// 	{
	// 		if (i == 0)
	// 			continue;
	// 		if (i < 0)
	// 		{
	// 			fDist = (i + 0.5) * fSliceDist;
	// 		}
	// 		else
	// 		{
	// 			fDist = (i - 0.5) * fSliceDist;
	// 		}
	// 	}
	// 	else
	// 	{
	// 		fDist = i * fSliceDist;
	// 	}

	// 	ptCenterSlice = ptCenterSlice + dirN * fDist;

	// 	Point3d ptLeftTop = ptCenterSlice - dirH * (0.5 * batchInfo.m_fLengthH);
	// 	ptLeftTop = ptLeftTop - dirV * (0.5 * batchInfo.m_fLengthV);

	// 	float3 ptLeftTop_cu;
	// 	ptLeftTop_cu.x = ptLeftTop[0];
	// 	ptLeftTop_cu.y = ptLeftTop[1];
	// 	ptLeftTop_cu.z = ptLeftTop[2];

	// 	short *pData = new short[nWidth * nHeight];
	// 	switch (batchInfo.m_MPRType)
	// 	{
	// 	case MPRTypeAverage:
	// 	{
	// 		cu_renderPlane_Average(
	// 			pData,
	// 			nWidth,
	// 			nHeight,
	// 			m_cuDataInfo.m_h_volumeTexture,
	// 			m_VolumeSize,
	// 			m_f3Spacing,
	// 			dirH_cu,
	// 			dirV_cu,
	// 			dirN_cu,
	// 			ptLeftTop_cu,
	// 			batchInfo.m_fPixelSpacing,
	// 			DataManager::Instance()->Need2InvertZ(),
	// 			halfNum);
	// 	}
	// 	break;
	// 	default:
	// 		return false;
	// 	}
	// 	vecBatchData.push_back(pData);
	// }
	return true;
}

void Render::ShowPlaneInVR(bool bShow)
{
	IRender::ShowPlaneInVR(bShow);
	// m_cuDataInfo.CopyMaskData(DataManager::Instance()->GetMaskData().get(), m_VolumeSize);
}
