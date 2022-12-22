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

    return true;
}