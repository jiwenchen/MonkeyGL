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

#include "CPRImageLayer.h"
#include <driver_types.h>
#include "vector_types.h"
#include "DataManager.h"

using namespace MonkeyGL;

extern "C" void cu_renderCPR(
	short *pData,
	int width,
	int height,
	cudaTextureObject_t volumeTexture,
	cudaExtent volumeSize,
	double *pPoints,
	double *pDirs,
	bool invertZ);

CPRImageLayer::CPRImageLayer()
: ImageLayer()
{
}

CPRImageLayer::~CPRImageLayer()
{
}

bool CPRImageLayer::GetGrayscaleData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, PlaneType planeType)
{
    if (PlaneStretchedCPR != planeType && PlaneStraightenedCPR != planeType){
        return false;
    }

	Point3d *pPoints = NULL;
	Direction3d *pDirs = NULL;

	if (!DataManager::Instance()->GetCPRInfo(pPoints, pDirs, nWidth, nHeight, planeType))
		return false;

    pData.reset(new short[nWidth * nHeight]);

    cu_renderCPR(
		pData.get(),
		nWidth,
		nHeight,
		DataManager::Instance()->GetCuDataManager().GetVolumeTexture(),
		DataManager::Instance()->GetVolumeSize(),
		(double *)pPoints,
		(double *)pDirs,
		DataManager::Instance()->Need2InvertZ());

	delete[] pPoints;
	delete[] pDirs;

    return false;
}