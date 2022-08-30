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


