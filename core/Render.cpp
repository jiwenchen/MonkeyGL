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
#include "DataManager.h"

using namespace MonkeyGL;

extern "C" void cu_test_3d(short *h_volumeData, cudaExtent volumeSize);
extern "C" void cu_init_id();
extern "C" bool cu_set_1d(float *h_volumeData, int nLen, unsigned char nLabel);
extern "C" void cu_test_1d(int nLen, unsigned char nLabel);

Render::Render(void)
{
	// testcuda();
}

void Render::testcuda()
{
#if 1
	int nWidth = 512;
	int nHeight = 512;
	int nDepth = 200;
	short *pData = new short[nWidth * nHeight * nDepth];
	for (int i = 0; i < nDepth; i++)
	{
		for (int j = 0; j < nWidth * nHeight; j++)
		{
			pData[i * nHeight * nWidth + j] = j + i;
		}
	}
	// m_VolumeSize.width = nWidth;
	// m_VolumeSize.height = nHeight;
	// m_VolumeSize.depth = nDepth;

	// cu_test_3d(pData, m_VolumeSize);

	delete[] pData;
#else
	cu_init_id();
	int nLen = 100;
	float *pData = new float[nLen * 4];
	for (int i = 0; i < nLen; i++)
	{
		pData[4 * i] = i;
		pData[4 * i + 1] = i;
		pData[4 * i + 2] = i;
		pData[4 * i + 3] = i + 1;
	}
	cu_set_1d(pData, nLen, 1);

	for (int i = 0; i < nLen; i++)
	{
		pData[4 * i] = i + 20;
		pData[4 * i + 1] = i + 20;
		pData[4 * i + 2] = i + 20;
		pData[4 * i + 3] = i + 20 + 1;
	}
	cu_set_1d(pData, nLen, 10);

	for (int l = 0; l < 10000; l++)
	{
		for (int i = 0; i < 12; i++)
			cu_test_1d(nLen, i);
	}

	delete[] pData;
#endif
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

void Render::MergeOrientationBox(unsigned char *pVR, int nWidth, int nHeight)
{
	StopWatch sw("Render::MergeOrientationBox");
	int nW_Box = 64, nH_Box = 64;
	if (nWidth <= nW_Box || nHeight <= nH_Box)
	{
		return;
	}

	{
		int w = 512;
		int h = 512;
		std::vector<Facet2D> facet2Ds = GetMeshPoints(w, h);
		std::shared_ptr<float> pImage(new float[w * h]);
		memset(pImage.get(), 0, w * h * sizeof(float));
		std::shared_ptr<float> pZBuffer(new float[w * h]);
		memset(pZBuffer.get(), 0, w * h * sizeof(float));

		for (size_t i = 0; i < facet2Ds.size(); i++)
		{
			Facet2D &facet2D = facet2Ds[i];

			Point2f &v1 = facet2D.v1;
			Point2f &v2 = facet2D.v2;
			Point2f &v3 = facet2D.v3;
			float &zBuffer = facet2D.zBuffer;
			Methods::FillHoleInImage_Ch1(pImage.get(), pZBuffer.get(), w, h, facet2D.diffuse, zBuffer, v1, v2, v3);
		}

		RGB clr(0.902, 0.902, 0.302);
		for (int y = nHeight - nH_Box; y < nHeight; y++)
		{
			int yIdx = (y + nH_Box - nHeight) * 8;
			for (int x = nWidth - nW_Box; x < nWidth; x++)
			{
				int xIdx = (x + nW_Box - nWidth) * 8;
				float diffuse = pImage.get()[yIdx * w + xIdx];
				if (diffuse > 0)
				{
					RGB clrTemp = clr * diffuse;
					clrTemp = clrTemp + 0.05;
					int red = int(clrTemp.red * 255);
					int green = int(clrTemp.green * 255);
					int blue = int(clrTemp.blue * 255);
					pVR[3 * (y * nWidth + x)] = red;
					pVR[3 * (y * nWidth + x) + 1] = green;
					pVR[3 * (y * nWidth + x) + 2] = blue;
				}
			}
		}
	}
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

Point2f TransferPoint2D(Point2f pt, int nWidth, int nHeight)
{
	float r = nHeight;
	Point2f ptOut = pt;
	ptOut *= r;
	ptOut += Point2f(nWidth / 2.0, nHeight / 2.0);
	return ptOut;
}

std::vector<Facet2D> Render::GetMeshPoints(int nWidth, int nHeight)
{
	std::vector<Facet3D> facet3Ds = m_marchingCube.GetMesh();
	std::vector<Facet2D> facet2Ds;
	// for (size_t i = 0; i < facet3Ds.size(); i++)
	// {
	// 	Facet3D &facet3D = facet3Ds[i];

	// 	Point3f v1 = Methods::GetTransferPointf(m_pRotateMatrix, facet3D.v1);
	// 	Point3f v2 = Methods::GetTransferPointf(m_pRotateMatrix, facet3D.v2);
	// 	Point3f v3 = Methods::GetTransferPointf(m_pRotateMatrix, facet3D.v3);

	// 	Direction3f n = Direction3f(v3, v1).cross(Direction3f(v3, v2));
	// 	float d = n.dot(Direction3f(0, -1, 0));
	// 	if (d < 0)
	// 	{
	// 		continue;
	// 	}

	// 	Facet2D facet2D;
	// 	facet2D.diffuse = d;
	// 	facet2D.zBuffer = (v1.y() + v2.y() + v3.y()) / 3 + 10000;
	// 	facet2D.v1 = TransferPoint2D(Point2f(v1.x(), v1.z()), nWidth, nHeight);
	// 	facet2D.v2 = TransferPoint2D(Point2f(v2.x(), v2.z()), nWidth, nHeight);
	// 	facet2D.v3 = TransferPoint2D(Point2f(v3.x(), v3.z()), nWidth, nHeight);
	// 	facet2Ds.push_back(facet2D);
	// }
	return facet2Ds;
}