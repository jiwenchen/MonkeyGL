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

    class CuDataManager
    {
    public:
        CuDataManager();
        ~CuDataManager();

    public:
        void CopyVolumeData(short* h_volumeData, cudaExtent volumeSize);
        void CopyMaskData(unsigned char* h_maskData, cudaExtent volumeSize);
        void CopyOperatorMatrix( float *pTransformMatrix, float *pTransposeTransformMatrix);
        void SetTransferFunction( float* pTransferFunc, int nLenTransferFunc, unsigned char nLabel);

        const cudaTextureObject_t& GetVolumeTexture(){
            return m_h_volumeTexture;
        }

        const cudaTextureObject_t& GetMaskTexture(){
            return m_h_maskTexture;
        }

        const cudaTextureObjects& GetTransferFuncTextures(){
            return m_h_transferFuncTextures;
        }

        const float3x3& GetTransformMatrix(){
            return m_h_transformMatrix;
        }

    private:
        friend class DataManager;

        cudaTextureObject_t m_h_volumeTexture;
        cudaArray* m_d_volumeArray;

        cudaTextureObject_t m_h_maskTexture;
        cudaArray* m_d_maskArray;

        float3x3 m_h_transformMatrix;

        cudaTextureObjects m_h_transferFuncTextures;
        cudaArray* m_d_transferFuncArrays[MAXOBJECTCOUNT+1];
    };
}