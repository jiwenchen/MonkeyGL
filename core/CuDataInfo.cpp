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

#include "CuDataInfo.h"

using namespace MonkeyGL;

extern "C"
void cu_copyVolumeData(short* h_volumeData, cudaExtent volumeSize, cudaArray*& d_volumeArray, cudaTextureObject_t& volumeTexture);
extern "C"
void cu_copyMaskData(unsigned char* h_maskData, cudaExtent volumeSize, cudaArray*& d_maskArray, cudaTextureObject_t& maskTexture);
extern "C"
bool cu_setTransferFunc( float* pTransferFunc, int nLenTransferFunc, cudaArray*& d_transferFuncArray, cudaTextureObject_t& transferFuncTexture);

CuDataInfo::CuDataInfo()
{
    m_d_volumeArray = 0;
    m_d_maskArray = 0;
    for (int i=0; i<MAXOBJECTCOUNT; i++){
		m_d_transferFuncArrays[i] = 0;
	}
}

CuDataInfo::~CuDataInfo()
{
    if (m_d_volumeArray != 0)
	{
		checkCudaErrors(cudaFreeArray(m_d_volumeArray));
		m_d_volumeArray = 0;
	}
	if (m_d_maskArray != 0)
	{
		checkCudaErrors(cudaFreeArray(m_d_maskArray));
		m_d_maskArray = 0;
	}
    for (int i=0; i<MAXOBJECTCOUNT; i++){
		if (m_d_transferFuncArrays[i] != 0)
		{
			checkCudaErrors(cudaFreeArray(m_d_transferFuncArrays[i]));
			m_d_transferFuncArrays[i] = 0;
		}
	}
}

void CuDataInfo::CopyVolumeData(short* h_volumeData, cudaExtent volumeSize)
{
    cu_copyVolumeData(h_volumeData, volumeSize, m_d_volumeArray, m_h_volumeTexture);
}

void CuDataInfo::CopyMaskData(unsigned char* h_maskData, cudaExtent volumeSize)
{
    cu_copyMaskData(h_maskData, volumeSize, m_d_maskArray, m_h_maskTexture);
}

void CuDataInfo::SetTransferFunction( float* pTransferFunc, int nLenTransferFunc, unsigned char nLabel)
{
    cu_setTransferFunc(pTransferFunc, nLenTransferFunc, m_d_transferFuncArrays[nLabel], m_h_transferFuncTextures[nLabel]);
}

void CuDataInfo::CopyOperatorMatrix( float *pTransformMatrix, float *pTransposeTransformMatrix)
{
    memcpy(&m_h_transformMatrix, pTransformMatrix, sizeof(float3x3));
}


