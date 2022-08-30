#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "Defines.h"

namespace MonkeyGL{

    typedef struct {
        float3 m[3];
    } float3x3;

    typedef struct {
        cudaTextureObject_t m[MAXOBJECTCOUNT+1];
        cudaTextureObject_t& operator[] (int idx){
            return m[idx];
        };
    } cudaTextureObjects;

    class CuDataInfo
    {
    public:
        CuDataInfo();
        ~CuDataInfo();

    public:
        void CopyVolumeData(short* h_volumeData, cudaExtent volumeSize);
        void CopyMaskData(unsigned char* h_maskData, cudaExtent volumeSize);
        void CopyOperatorMatrix( float *pTransformMatrix, float *pTransposeTransformMatrix);
        void SetTransferFunction( float* pTransferFunc, int nLenTransferFunc, unsigned char nLabel);

    public:
        cudaTextureObject_t m_h_volumeTexture;
        cudaArray* m_d_volumeArray;

        cudaTextureObject_t m_h_maskTexture;
        cudaArray* m_d_maskArray;

        float3x3 m_h_transformMatrix;

        cudaTextureObjects m_h_transferFuncTextures;
        cudaArray* m_d_transferFuncArrays[MAXOBJECTCOUNT+1];
    };
}