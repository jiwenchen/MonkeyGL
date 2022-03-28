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

cudaArray *d_transferFuncArray_test = 0;
cudaTextureObject_t transferTex_test;


__global__ void transformKernel_1d(
    float* output,
    cudaTextureObject_t texObj,
    int nLen
) 
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    float4 c = tex1D<float4>(texObj, 1.0*x/nLen);
    output[4*x] = c.x;
    output[4*x+1] = c.y;
    output[4*x+2] = c.z;
    output[4*x+3] = c.w;
}


cudaResourceDesc texRes;
cudaTextureDesc texDescr;

extern "C"
void cu_init_test_1d()
{
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;

    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    // checkCudaErrors(
    //     cudaCreateTextureObject(&transferTex_test, &texRes, &texDescr, NULL)
    // );
}


extern "C" 
void cu_test_1d( float* pTransferFunc, int nLenTransferFunc )
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    checkCudaErrors(cudaMallocArray( &d_transferFuncArray_test, &channelDesc, nLenTransferFunc, 1));
    checkCudaErrors(
        cudaMemcpy2DToArray(
            d_transferFuncArray_test, 
            0, 
            0, 
            pTransferFunc,
            0, 
            nLenTransferFunc*sizeof(float4), 
            1,
            cudaMemcpyHostToDevice
        )
    );

    // cudaResourceDesc texRes;
    // memset(&texRes, 0, sizeof(cudaResourceDesc));

    // texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_transferFuncArray_test;

    // cudaTextureDesc texDescr;
    // memset(&texDescr, 0, sizeof(cudaTextureDesc));

    // texDescr.normalizedCoords = true;
    // texDescr.filterMode = cudaFilterModeLinear;
    // texDescr.addressMode[0] = cudaAddressModeClamp;
    // texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(
        cudaCreateTextureObject(&transferTex_test, &texRes, &texDescr, NULL)
    );

    float* result_arr;
    cudaMalloc(&result_arr, nLenTransferFunc * 4 * sizeof(float));

    dim3 blockSize(16);
	dim3 gridSize( (nLenTransferFunc-1)/blockSize.x+1 );
    transformKernel_1d<<<gridSize, blockSize>>>(result_arr, transferTex_test, nLenTransferFunc);

    float* pOut = (float*)malloc(nLenTransferFunc * 4 * sizeof(float));

    cudaMemcpy( pOut, result_arr, nLenTransferFunc * 4 * sizeof(float), cudaMemcpyDeviceToHost );

    for (int i=0; i<nLenTransferFunc; i++){
        printf("%.2f %.2f %.2f %.2f\n", pOut[4*i], pOut[4*i+1], pOut[4*i+2], pOut[4*i+3]);
    }
    
    free(pOut);
}

