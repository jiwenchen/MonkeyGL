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

#include "VRImageLayer.h"
#include <driver_types.h>
#include "vector_types.h"
#include "DataManager.h"
#include "Methods.h"
#include "StopWatch.h"

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
	VOI voi, 
	RGBA colorBG,
	RenderType type
);

VRImageLayer::VRImageLayer()
: ImageLayer()
{
}

VRImageLayer::~VRImageLayer()
{
}

bool VRImageLayer::GetRGBData(std::shared_ptr<unsigned char>& pData, int& nWidth, int& nHeight, PlaneType planeType)
{
	if (!DataManager::Instance()->IsLayerEnable(GetLayerType())){
		return false;
	}

    if (PlaneVR != planeType){
        return false;
    }

    DataManager::Instance()->GetRenderInfo().GetVRSize(nWidth, nHeight);
    pData.reset(new unsigned char[nWidth*nHeight*3]);

    cu_render(
		pData.get(), 
		nWidth, 
		nHeight, 
		DataManager::Instance()->GetCuDataManager().GetVolumeTexture(),
		DataManager::Instance()->GetVolumeSize(),
		DataManager::Instance()->GetCuDataManager().GetMaskTexture(),
		DataManager::Instance()->GetCuDataManager().GetTransferFuncTextures(),
		DataManager::Instance()->GetAlphaAndWWWLInfo(),
		DataManager::Instance()->GetMaxLenSpacing(),
		DataManager::Instance()->GetSpacing(),
		DataManager::Instance()->GetSpacingVoxel(),
		DataManager::Instance()->GetRenderInfo().GetTotalXTranslate(),
		DataManager::Instance()->GetRenderInfo().GetTotalYTranslate(),
		DataManager::Instance()->GetRenderInfo().GetTotalScale(),
		DataManager::Instance()->GetCuDataManager().GetTransformMatrix(),
		DataManager::Instance()->Need2InvertZ(),
		DataManager::Instance()->GetVOINormalize(),
		DataManager::Instance()->GetColorBackground(),
		DataManager::Instance()->GetRenderType()
	);

	MergeOrientationBox(pData.get(), nWidth, nHeight);

    return true;
}

void VRImageLayer::MergeOrientationBox(unsigned char *pVR, int nWidth, int nHeight)
{
	StopWatch sw("VRImageLayer::MergeOrientationBox");
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

Point2f TransferPoint2D(Point2f pt, int nWidth, int nHeight)
{
	float r = nHeight;
	Point2f ptOut = pt;
	ptOut *= r;
	ptOut += Point2f(nWidth / 2.0, nHeight / 2.0);
	return ptOut;
}

std::vector<Facet2D> VRImageLayer::GetMeshPoints(int nWidth, int nHeight)
{
	std::vector<Facet3D> facet3Ds = m_marchingCube.GetMesh();
	std::vector<Facet2D> facet2Ds;
	for (size_t i = 0; i < facet3Ds.size(); i++)
	{
		Facet3D &facet3D = facet3Ds[i];

		Point3f v1 = Methods::GetTransferPointf(DataManager::Instance()->GetRenderInfo().GetRotateMatrix(), facet3D.v1);
		Point3f v2 = Methods::GetTransferPointf(DataManager::Instance()->GetRenderInfo().GetRotateMatrix(), facet3D.v2);
		Point3f v3 = Methods::GetTransferPointf(DataManager::Instance()->GetRenderInfo().GetRotateMatrix(), facet3D.v3);

		Direction3f n = Direction3f(v3, v1).cross(Direction3f(v3, v2));
		float d = n.dot(Direction3f(0, -1, 0));
		if (d < 0)
		{
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