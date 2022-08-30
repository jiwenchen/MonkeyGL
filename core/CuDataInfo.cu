#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


extern "C"
void cu_copyVolumeData( short* h_volumeData, cudaExtent volumeSize, cudaArray*& d_volumeArray, cudaTextureObject_t& volumeTexture)
{
	if (d_volumeArray != 0)
	{
		checkCudaErrors(cudaFreeArray(d_volumeArray));
		d_volumeArray = 0;
		volumeTexture = 0;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<short>();
	checkCudaErrors( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

	cudaMemcpy3DParms copyParams = {0};
	copyParams.dstArray = d_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	copyParams.srcPtr   = make_cudaPitchedPtr(
		(void*)h_volumeData,
		volumeSize.width*sizeof(short),
		volumeSize.width,
		volumeSize.height
	);

	checkCudaErrors( cudaMemcpy3D(&copyParams) );  
	
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_volumeArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = true;  // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear;  // linear interpolation

	texDescr.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;
		
	checkCudaErrors( cudaCreateTextureObject(&volumeTexture, &texRes, &texDescr, NULL) );
}

extern "C"
void cu_copyMaskData(unsigned char* h_maskData, cudaExtent volumeSize, cudaArray*& d_maskArray, cudaTextureObject_t& maskTexture)
{
	if (d_maskArray != 0)
	{
		checkCudaErrors(cudaFreeArray(d_maskArray));
		d_maskArray = 0;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	checkCudaErrors( cudaMalloc3DArray(&d_maskArray, &channelDesc, volumeSize) );

	cudaMemcpy3DParms copyParams = {0};
	copyParams.dstArray = d_maskArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	copyParams.srcPtr   = make_cudaPitchedPtr(
		(void*)h_maskData,
		volumeSize.width*sizeof(unsigned char),
		volumeSize.width,
		volumeSize.height
	);

	checkCudaErrors( cudaMemcpy3D(&copyParams) );  
	
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_maskArray;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint; 

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeElementType;
		
	checkCudaErrors( cudaCreateTextureObject(&maskTexture, &texRes, &texDescr, NULL) );
}

extern "C"
void cu_setTransferFunc( float* pTransferFunc, int nLenTransferFunc, cudaArray*& d_transferFuncArray, cudaTextureObject_t& transferFuncTexture)
{
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

    if (d_transferFuncArray != 0)
	{
		checkCudaErrors(cudaFreeArray(d_transferFuncArray));
		d_transferFuncArray = 0;
	}
    checkCudaErrors(cudaMallocArray( &d_transferFuncArray, &channelDesc, nLenTransferFunc, 1));
    checkCudaErrors(
        cudaMemcpy2DToArray(
            d_transferFuncArray, 
            0, 
            0, 
            pTransferFunc,
            0, 
            nLenTransferFunc*sizeof(float4), 
            1,
            cudaMemcpyHostToDevice
        )
    );

    texRes.res.array.array = d_transferFuncArray;

    checkCudaErrors(
        cudaCreateTextureObject(&transferFuncTexture, &texRes, &texDescr, NULL)
    );
}