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

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

cudaArray *d_volumeArray_test = 0;
cudaTextureObject_t texObject_test;

typedef short VolumeType;

__global__ void transformKernel(float* output,
                                cudaTextureObject_t texObj,
                                int width, int height) 
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    output[y * width + x] = tex3D<float>(texObj, 1.0*x/width, 0, 0)*32768;
}

extern "C" 
void cu_test_3d( VolumeType* h_volumeData, cudaExtent volumeSize)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray_test, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(
        h_volumeData,
        volumeSize.width * sizeof(VolumeType),
        volumeSize.width, 
        volumeSize.height
    );
    copyParams.dstArray = d_volumeArray_test;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_volumeArray_test;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    // texDescr.filterMode = cudaFilterModePoint;
    texDescr.filterMode = cudaFilterModeLinear;

    texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    // texDescr.readMode = cudaReadModeElementType;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(
        cudaCreateTextureObject(&texObject_test, &texRes, &texDescr, NULL)
    );

    float* result_arr;
    cudaMalloc(&result_arr, volumeSize.width * volumeSize.height * sizeof(float));

    dim3 blockSize(16, 16);
	dim3 gridSize( (volumeSize.width-1)/blockSize.x+1, (volumeSize.height-1)/blockSize.y+1 );
    transformKernel<<<gridSize, blockSize>>>(result_arr, texObject_test, volumeSize.width, volumeSize.height);

    float* pOut = (float*)malloc(volumeSize.width * volumeSize.height * sizeof(float));

    cudaMemcpy( pOut, result_arr, volumeSize.width * volumeSize.height * sizeof(VolumeType), cudaMemcpyDeviceToHost );

    for (int i=0; i<volumeSize.width; i++){
        printf("%.0f ", pOut[i]);
    }
    printf("\n\n");
    for (int i=volumeSize.width; i<volumeSize.width*2; i++){
        printf("%.2f ", pOut[i]);
    }

    free(pOut);
}

__constant__ cudaTextureObject_t cTexts[32];
cudaTextureObject_t texts[32];
cudaArray *d_transferFuncArray_test[32];


extern "C"
void cu_init_id()
{
    for (int i=0; i<32; i++){
        d_transferFuncArray_test[i] = 0;
    }
}

extern "C" 
bool cu_set_1d( float* pTransferFunc, int nLenTransferFunc, unsigned char nLabel )
{
    if (nLabel >= 32){
        return false;
    }
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    if (d_transferFuncArray_test[nLabel] != 0)
	{
		checkCudaErrors(cudaFreeArray(d_transferFuncArray_test[nLabel]));
		d_transferFuncArray_test[nLabel] = 0;
	}
    checkCudaErrors(cudaMallocArray( &d_transferFuncArray_test[nLabel], &channelDesc, nLenTransferFunc, 1));
    checkCudaErrors(
        cudaMemcpy2DToArray(
            d_transferFuncArray_test[nLabel], 
            0, 
            0, 
            pTransferFunc,
            0, 
            nLenTransferFunc*sizeof(float4), 
            1,
            cudaMemcpyHostToDevice
        )
    );

    texRes.res.array.array = d_transferFuncArray_test[nLabel];

    cudaTextureObject_t text = 0;
    checkCudaErrors(
        cudaCreateTextureObject(&text, &texRes, &texDescr, NULL)
    );

    texts[nLabel] = text;
    cudaMemcpyToSymbol(cTexts, texts, sizeof(texts));

    return true;
}

__global__ void transformKernel_1d(
    float* output,
    int nLen,
    unsigned char nLabel
) 
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    float4 c = tex1D<float4>(cTexts[nLabel], 1.0*x/nLen);
    output[4*x] = c.x;
    output[4*x+1] = c.y;
    output[4*x+2] = c.z;
    output[4*x+3] = c.w;
}

extern "C" 
void cu_test_1d(int nLenTransferFunc, unsigned char nLabel)
{
    float* result_arr;
    cudaMalloc(&result_arr, nLenTransferFunc * 4 * sizeof(float));

    dim3 blockSize(16);
	dim3 gridSize( (nLenTransferFunc-1)/blockSize.x+1 );
    transformKernel_1d<<<gridSize, blockSize>>>(result_arr, nLenTransferFunc, nLabel);

    float* pOut = (float*)malloc(nLenTransferFunc * 4 * sizeof(float));

    cudaMemcpy( pOut, result_arr, nLenTransferFunc * 4 * sizeof(float), cudaMemcpyDeviceToHost );
    cudaFree(result_arr);

    for (int i=0; i<nLenTransferFunc; i++){
        printf("%.2f %.2f %.2f %.2f\n", pOut[4*i], pOut[4*i+1], pOut[4*i+2], pOut[4*i+3]);
    }
    
    free(pOut);
}

