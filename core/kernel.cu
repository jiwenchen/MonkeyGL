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
#include "Defines.h"
#include "CuDataInfo.h"

using namespace MonkeyGL;

__device__ float3 mul(const float3x3 &M, const float3 &v)
{
	float3 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = 0.0f;
	return (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ float4 tracing(
	float4 sum,
	float alphaAccObject,
	cudaTextureObject_t volumeTexture,
	float3 pos,
	float4 col,
	float3 dirLight,
	float3 f3SpacingVoxel,
	bool invertZ
)
{
	float3 N;
	N.x = tex3D<float>(volumeTexture, pos.x+f3SpacingVoxel.x, pos.y, pos.z) - tex3D<float>(volumeTexture, pos.x-f3SpacingVoxel.x, pos.y, pos.z);
	N.y = tex3D<float>(volumeTexture, pos.x, pos.y+f3SpacingVoxel.y, pos.z) - tex3D<float>(volumeTexture, pos.x, pos.y-f3SpacingVoxel.y, pos.z);
	N.z = tex3D<float>(volumeTexture, pos.x, pos.y, pos.z+f3SpacingVoxel.z) - tex3D<float>(volumeTexture, pos.x, pos.y, pos.z-f3SpacingVoxel.z);
	if (invertZ){
		N.z = -N.z;
	}
	N = normalize(N);

	float diffuse = dot(N, dirLight);
	float4 clrLight = col * 0.35f;

	float4 f4Temp = make_float4(0.0f);
	if ( diffuse > 0.0f )
	{
		f4Temp = col * (diffuse*0.8f + 0.16f*(pow(diffuse, 8.0f)));
	}
	clrLight += f4Temp;

	diffuse = (1.0f - alphaAccObject) * col.w;
	return (sum + diffuse * clrLight);
}


__device__ bool getNextStep(
	float& fAlphaTemp,
	float& fStepTemp,
	float& accuLength,
	float fAlphaPre,
	float fStepL1,
	float fStepL4,
	float fStepL8
)
{
	if (fStepTemp == fStepL4)
		fAlphaTemp = 1 - pow(1-fAlphaTemp, 0.25f);
	else if(fStepTemp == fStepL8)
		fAlphaTemp = 1 - pow(1-fAlphaTemp, 0.125f);

	if (accuLength > 0.0f)
	{	
		if (MAX(fAlphaTemp, fAlphaPre) > 0.001f)
		{					
			if (fStepTemp == fStepL1)
			{		
				accuLength -= (fStepL1 - fStepL4);
				fStepTemp = fStepL4;		
				return false;
			}
			else if(fStepTemp == fStepL4)
			{
				accuLength -= (fStepL4 - fStepL8);
				fStepTemp = fStepL8;		
				return false;
			}
		}
		else
		{
			if (fStepTemp == fStepL8)
				fStepTemp = fStepL4;
			else
				fStepTemp = fStepL1;
		}
	}
	return true;
}

/*
**   z
**   |__x
**  /-y
*/

__global__ void d_render(
	unsigned char* pPixelData,
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
	float4 f4ColorBG
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		float u = 1.0f*(x-width/2.0f-xTranslate)/height;
		float v = 1.0f*(y-height/2.0f-yTranslate)/height;

		float4 sum = make_float4(0.0f);

		float3 dirLight = make_float3(0.0f, 1.0f, 0.0f);
		dirLight = normalize(mul(transformMatrix, dirLight));

		float fStepL1 = 1.0f/volumeSize.depth;
		float fStepL4 = fStepL1/4.0f;
		float fStepL8 = fStepL1/8.0f;
		float fStepTemp = fStepL1;

		float temp = 0.0f;
		float3 pos;

		float alphaAccObject[MAXOBJECTCOUNT+1];
		for (int i=0; i<MAXOBJECTCOUNT+1; i++){
			alphaAccObject[i] = 0.0f;
		}
		float alphaAcc = 0.0f;

		float accuLength = 0.0f;
		int nxIdx = 0;
		int nyIdx = 0;
		int nzIdx = 0;
		float fy = 0;

		float4 col;
		float fAlphaTemp = 0.0f;
		float fAlphaPre = 0.0f;

		unsigned char label = 0;

		while (accuLength < 1.732)
		{
			fy = (accuLength-0.866)*scale;

			pos = make_float3(u, fy, v);
			pos = mul(transformMatrix, pos);

			pos.x = pos.x * f3maxLenSpacing.x + 0.5f;
			pos.y = pos.y * f3maxLenSpacing.y + 0.5f;
			pos.z = pos.z * f3maxLenSpacing.z + 0.5f;
			if (invertZ)
				pos.z = 1.0f - pos.z;

			nxIdx = pos.x * volumeSize.width;
			nyIdx = pos.y * volumeSize.height;
			nzIdx = pos.z * volumeSize.depth;
			if (nxIdx<voi.left || nxIdx>voi.right || nyIdx<voi.posterior || nyIdx>voi.anterior || nzIdx<voi.head || nzIdx>voi.foot)
			{
				accuLength += fStepTemp;
				continue;
			}
			if(maskTexture == 0){
				label = 0;
			}
			else {
				label = tex3D<unsigned char>(maskTexture, nxIdx, nyIdx, nzIdx);
			}

			temp = 32768*tex3D<float>(volumeTexture, pos.x, pos.y, pos.z);
			temp = (temp - alphaAndWWWLInfo.m[label].wl)/alphaAndWWWLInfo.m[label].ww + 0.5;

			if (temp>1)
				temp = 1;

			col = tex1D<float4>(transferFuncTextures.m[label], temp);

			fAlphaTemp = col.w;

			if (!getNextStep(fAlphaTemp, fStepTemp, accuLength, fAlphaPre, fStepL1, fStepL4, fStepL8)){
				continue;
			}	
			
			fAlphaPre = fAlphaTemp;
			accuLength += fStepTemp;

			col.w = fAlphaTemp;

			if (col.w > 0.0005f && alphaAccObject[label] < alphaAndWWWLInfo.m[label].alpha){
				sum = tracing(sum, alphaAcc, volumeTexture, pos, col, dirLight, f3SpacingVoxel, invertZ);
				alphaAccObject[label] += (1.0f - alphaAcc) * col.w;
				alphaAcc += (1.0f - alphaAcc) * col.w;
			}

			if (alphaAcc > 0.995f){
				break;
			}

		}
		
		if (sum.x==0.0f && sum.y==0.0f && sum.z==0.0f && sum.w==0.0f){
			sum = f4ColorBG;
		}

		unsigned int result = rgbaFloatToInt(sum);

		pPixelData[nIdx*3]	 = result & 0xFF; //R
		pPixelData[nIdx*3+1] = (result>>8) & 0xFF; //G
		pPixelData[nIdx*3+2] = (result>>16) & 0xFF; //B
	}
}

__global__ void d_renderMIP(
	unsigned char* pPixelData,
	int width,
	int height,
	cudaTextureObject_t volumeTexture,
	cudaExtent volumeSize,
	cudaTextureObject_t maskTexture,
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
	float4 f4ColorBG
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		float u = 1.0f*(x-width/2.0f-xTranslate)/height;
		float v = 1.0f*(y-height/2.0f-yTranslate)/height;

		float4 sum = make_float4(0.0f);
		float fStep = 1.0f/volumeSize.depth;
		float temp = 0.0f;
		float3 pos;

		float alphaAcc = 0.0f;
		float accuLength = 0.0f;

		int nxIdx = 0;
		int nyIdx = 0;
		int nzIdx = 0;
		float fy = 0;

		unsigned char label = 0;

		while (accuLength < 1.732)
		{
			fy = (accuLength-0.866)*scale;

			pos = make_float3(u, fy, v);
			pos = mul(transformMatrix, pos);

			pos.x = pos.x * f3maxLenSpacing.x + 0.5f;
			pos.y = pos.y * f3maxLenSpacing.y + 0.5f;
			pos.z = pos.z * f3maxLenSpacing.z + 0.5f;
			if (invertZ)
				pos.z = 1.0f - pos.z;

			nxIdx = pos.x * volumeSize.width;
			nyIdx = pos.y * volumeSize.height;
			nzIdx = pos.z * volumeSize.depth;
			if (nxIdx<voi.left || nxIdx>voi.right || nyIdx<voi.posterior || nyIdx>voi.anterior || nzIdx<voi.head || nzIdx>voi.foot)
			{
				accuLength += fStep;
				continue;
			}
			if(maskTexture == 0){
				label = 0;
			}
			else {
				label = tex3D<unsigned char>(maskTexture, nxIdx, nyIdx, nzIdx);
			}

			if (alphaAndWWWLInfo.m[label].alpha > 0){
				temp = 32768*tex3D<float>(volumeTexture, pos.x, pos.y, pos.z);
				temp = (temp - alphaAndWWWLInfo.m[label].wl)/alphaAndWWWLInfo.m[label].ww + 0.5;	

				if (alphaAcc < temp){
					alphaAcc = temp;
				}
			}
			
			accuLength += fStep;
		}
		
		if (alphaAcc <= 0.0f){
			sum = f4ColorBG;
		}
		else{
			sum = make_float4(alphaAcc);
		}

		unsigned int result = rgbaFloatToInt(sum);

		pPixelData[nIdx*3]	 = result & 0xFF; //R
		pPixelData[nIdx*3+1] = (result>>8) & 0xFF; //G
		pPixelData[nIdx*3+2] = (result>>16) & 0xFF; //B
	}
}

__global__ void d_renderSurface(
	unsigned char* pPixelData,
	int width,
	int height,
	cudaTextureObject_t volumeTexture,
	cudaExtent volumeSize,
	cudaTextureObject_t maskTexture,
	float3 f3maxLenSpacing,
	float3 f3Spacing,
	float3 f3SpacingVoxel,
	float xTranslate,
	float yTranslate,
	float scale,
	float3x3 transformMatrix,
	bool invertZ,
	VOI voi,
	float4 f4ColorBG
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		float u = 1.0f*(x-width/2.0f-xTranslate)/height;
		float v = 1.0f*(y-height/2.0f-yTranslate)/height;

		float4 sum = make_float4(0.0f);
		float fStep = 1.0f/volumeSize.depth;
		float temp = 0.0f;
		float3 pos;

		float alphaAcc = 0.0f;
		float accuLength = 0.0f;

		int nxIdx = 0;
		int nyIdx = 0;
		int nzIdx = 0;
		float fy = 0;

		float3 dirLight = make_float3(0.0f, 1.0f, 0.0f);
		dirLight = normalize(mul(transformMatrix, dirLight));

		float3 N;

		while (accuLength < 1.732)
		{
			fy = (accuLength-0.866)*scale;

			pos = make_float3(u, fy, v);
			pos = mul(transformMatrix, pos);

			pos.x = pos.x * f3maxLenSpacing.x + 0.5f;
			pos.y = pos.y * f3maxLenSpacing.y + 0.5f;
			pos.z = pos.z * f3maxLenSpacing.z + 0.5f;
			if (invertZ)
				pos.z = 1.0f - pos.z;

			nxIdx = pos.x * volumeSize.width;
			nyIdx = pos.y * volumeSize.height;
			nzIdx = pos.z * volumeSize.depth;
			if (nxIdx<voi.left || nxIdx>voi.right || nyIdx<voi.posterior || nyIdx>voi.anterior || nzIdx<voi.head || nzIdx>voi.foot)
			{
				accuLength += fStep;
				continue;
			}

			temp = 32768*tex3D<float>(volumeTexture, pos.x, pos.y, pos.z);
			if (temp <= 400){
				accuLength += fStep;
				continue;
			}

			N.x = tex3D<float>(volumeTexture, pos.x+f3SpacingVoxel.x, pos.y, pos.z) - tex3D<float>(volumeTexture, pos.x-f3SpacingVoxel.x, pos.y, pos.z);
			N.y = tex3D<float>(volumeTexture, pos.x, pos.y+f3SpacingVoxel.y, pos.z) - tex3D<float>(volumeTexture, pos.x, pos.y-f3SpacingVoxel.y, pos.z);
			N.z = tex3D<float>(volumeTexture, pos.x, pos.y, pos.z+f3SpacingVoxel.z) - tex3D<float>(volumeTexture, pos.x, pos.y, pos.z-f3SpacingVoxel.z);
			if (invertZ){
				N.z = -N.z;
			}
			N = normalize(N);
			alphaAcc = dot(N, dirLight);
			if (alphaAcc > 0){
				break;
			}
			accuLength += fStep;
		}
		
		if (alphaAcc <= 0.0f){
			sum = f4ColorBG;
		}
		else{
			// sum = make_float4(alphaAcc+0.05, 0.0f, 0.0f, 1.0f);
			sum = make_float4(alphaAcc+0.05, alphaAcc+0.05, alphaAcc+0.05, 1.0f);
		}

		unsigned int result = rgbaFloatToInt(sum);

		pPixelData[nIdx*3]	 = result & 0xFF; //R
		pPixelData[nIdx*3+1] = (result>>8) & 0xFF; //G
		pPixelData[nIdx*3+2] = (result>>16) & 0xFF; //B
	}
}

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
)
{
	unsigned char* d_pVR = 0;
	checkCudaErrors(cudaMalloc( (void**)&d_pVR, width*height*3*sizeof(unsigned char) ));

	dim3 blockSize(32, 32);
	dim3 gridSize( (width-1)/blockSize.x+1, (height-1)/blockSize.y+1 );

	float4 clrBG = make_float4(colorBG.red, colorBG.green, colorBG.blue, colorBG.alpha);

	if (RenderTypeVR == type){
		d_render<<<gridSize, blockSize>>>(
			d_pVR,
			width,
			height,
			volumeTexture,
			volumeSize,
			maskTexture,
			transferFuncTextures,
			alphaAndWWWLInfo,
			f3maxLenSpacing,
			f3Spacing,
			f3SpacingVoxel,
			xTranslate,
			yTranslate,
			scale,
			transformMatrix,
			invertZ,
			voi,
			clrBG
		);
	}
	else if (RenderTypeMIP == type){
		d_renderMIP<<<gridSize, blockSize>>>(
			d_pVR,
			width,
			height,
			volumeTexture,
			volumeSize,
			maskTexture,
			alphaAndWWWLInfo,
			f3maxLenSpacing,
			f3Spacing,
			f3SpacingVoxel,
			xTranslate,
			yTranslate,
			scale,
			transformMatrix,
			invertZ,
			voi,
			clrBG
		);
	}
	else if (RenderTypeSurface == type){
		d_renderSurface<<<gridSize, blockSize>>>(
			d_pVR,
			width,
			height,
			volumeTexture,
			volumeSize,
			maskTexture,
			f3maxLenSpacing,
			f3Spacing,
			f3SpacingVoxel,
			xTranslate,
			yTranslate,
			scale,
			transformMatrix,
			invertZ,
			voi,
			clrBG
		);
	}

	cudaError_t t = cudaMemcpy( pVR, d_pVR, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost );
	checkCudaErrors(cudaFree(d_pVR));
}

__global__ void d_renderPlane_MIP(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing, 
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		pData[nIdx] = -32768;
		short nVal = -32768;
		for (float t=-halfNum; t<=halfNum; t+=1)
		{
			float fLength = t*fPixelSpacing;
			float pt_x = ptLeftTop.x + fLength*dirN.x;
			float pt_y = ptLeftTop.y + fLength*dirN.y;
			float pt_z = ptLeftTop.z + fLength*dirN.z;
			float fx = (pt_x + x*fPixelSpacing*dirH.x + y*fPixelSpacing*dirV.x)/(f3Spacing.x*volumeSize.width);
			float fy = (pt_y + x*fPixelSpacing*dirH.y + y*fPixelSpacing*dirV.y)/(f3Spacing.y*volumeSize.height);
			float fz = (pt_z + x*fPixelSpacing*dirH.z + y*fPixelSpacing*dirV.z)/(f3Spacing.z*volumeSize.depth);
			if (!invertZ)
				fz = 1.0 - fz;

			if (fx>=0 && fx<=1 && fy>=0 && fy<=1 && fz>=0 && fz<=1)
				nVal = 32768*tex3D<float>(volumeTexture, fx, fy, fz);
			else
				nVal = -32768;
			pData[nIdx] = pData[nIdx]>nVal ? pData[nIdx]:nVal;
		}
	}
}

extern "C"
void cu_renderPlane_MIP(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing,
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
)
{
	short* d_pMPR = 0;
	checkCudaErrors(cudaMalloc( (void**)&d_pMPR, width*height*sizeof(short) ));
	checkCudaErrors( cudaMemset( d_pMPR, 0, width*height*sizeof(short) ) );

	dim3 blockSize(16, 16);
	dim3 gridSize( (width-1)/blockSize.x+1, (height-1)/blockSize.y+1 );

	d_renderPlane_MIP<<<gridSize, blockSize>>>(
		d_pMPR, 
		width, 
		height, 
		volumeTexture, 
		volumeSize, 
		f3Spacing, 
		dirH, 
		dirV, 
		dirN, 
		ptLeftTop, 
		fPixelSpacing, 
		invertZ, 
		halfNum
	);

	cudaError_t t = cudaMemcpy( pData, d_pMPR, width*height*sizeof(short), cudaMemcpyDeviceToHost );
	checkCudaErrors(cudaFree(d_pMPR));
}

__global__ void d_renderPlane_MinIP(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing, 
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		pData[nIdx] = 32767;
		short nVal = 32767;
		for (float t=-halfNum; t<=halfNum; t+=1)
		{
			float fLength = t*fPixelSpacing;
			float pt_x = ptLeftTop.x + fLength*dirN.x;
			float pt_y = ptLeftTop.y + fLength*dirN.y;
			float pt_z = ptLeftTop.z + fLength*dirN.z;
			float fx = (pt_x + x*fPixelSpacing*dirH.x + y*fPixelSpacing*dirV.x)/(f3Spacing.x*volumeSize.width);
			float fy = (pt_y + x*fPixelSpacing*dirH.y + y*fPixelSpacing*dirV.y)/(f3Spacing.y*volumeSize.height);
			float fz = (pt_z + x*fPixelSpacing*dirH.z + y*fPixelSpacing*dirV.z)/(f3Spacing.z*volumeSize.depth);
			if (!invertZ)
				fz = 1.0 - fz;

			if (fx>=0 && fx<=1 && fy>=0 && fy<=1 && fz>=0 && fz<=1)
				nVal = 32768*tex3D<float>(volumeTexture, fx, fy, fz);
			else
				nVal = -32768;
			pData[nIdx] = pData[nIdx]<nVal ? pData[nIdx]:nVal;
		}
	}
}

extern "C"
void cu_renderPlane_MinIP(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing,
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
)
{
	short* d_pMPR = 0;
	checkCudaErrors(cudaMalloc( (void**)&d_pMPR, width*height*sizeof(short) ));
	checkCudaErrors( cudaMemset( d_pMPR, 0, width*height*sizeof(short) ) );

	dim3 blockSize(16, 16);
	dim3 gridSize( (width-1)/blockSize.x+1, (height-1)/blockSize.y+1 );

	d_renderPlane_MinIP<<<gridSize, blockSize>>>(
		d_pMPR, 
		width, 
		height, 
		volumeTexture, 
		volumeSize, 
		f3Spacing, 
		dirH, 
		dirV, 
		dirN, 
		ptLeftTop, 
		fPixelSpacing, 
		invertZ, 
		halfNum
	);

	cudaError_t t = cudaMemcpy( pData, d_pMPR, width*height*sizeof(short), cudaMemcpyDeviceToHost );
	checkCudaErrors(cudaFree(d_pMPR));
}

__global__ void d_renderPlane_Average(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing, 
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		double fSum = 0;
		for (float t=-halfNum; t<=halfNum; t+=1)
		{
			float fLength = t*fPixelSpacing;
			float pt_x = ptLeftTop.x + fLength*dirN.x;
			float pt_y = ptLeftTop.y + fLength*dirN.y;
			float pt_z = ptLeftTop.z + fLength*dirN.z;
			float fx = (pt_x + x*fPixelSpacing*dirH.x + y*fPixelSpacing*dirV.x)/(f3Spacing.x*volumeSize.width);
			float fy = (pt_y + x*fPixelSpacing*dirH.y + y*fPixelSpacing*dirV.y)/(f3Spacing.y*volumeSize.height);
			float fz = (pt_z + x*fPixelSpacing*dirH.z + y*fPixelSpacing*dirV.z)/(f3Spacing.z*volumeSize.depth);
			if (!invertZ)
				fz = 1.0 - fz;

			if (fx>=0 && fx<=1 && fy>=0 && fy<=1 && fz>=0 && fz<=1)
				fSum += 32768*tex3D<float>(volumeTexture, fx, fy, fz);
			else
				fSum += -32768;
		}
		pData[nIdx] = fSum/(2*halfNum+1);
	}
}

extern "C"
void cu_renderPlane_Average(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	float3 f3Spacing,
	float3 dirH, 
	float3 dirV, 
	float3 dirN, 
	float3 ptLeftTop, 
	float fPixelSpacing, 
	bool invertZ, 
	float halfNum
)
{
	short* d_pMPR = 0;
	checkCudaErrors(cudaMalloc( (void**)&d_pMPR, width*height*sizeof(short) ));
	checkCudaErrors( cudaMemset( d_pMPR, 0, width*height*sizeof(short) ) );

	dim3 blockSize(16, 16);
	dim3 gridSize( (width-1)/blockSize.x+1, (height-1)/blockSize.y+1 );

	d_renderPlane_Average<<<gridSize, blockSize>>>(
		d_pMPR, 
		width, 
		height, 
		volumeTexture, 
		volumeSize, 
		f3Spacing, 
		dirH, 
		dirV, 
		dirN, 
		ptLeftTop, 
		fPixelSpacing, 
		invertZ, 
		halfNum
	);

	cudaError_t t = cudaMemcpy( pData, d_pMPR, width*height*sizeof(short), cudaMemcpyDeviceToHost );
	checkCudaErrors(cudaFree(d_pMPR));
}

__global__ void d_renderCPR(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	double* pPoints, 
	double* pDirs, 
	bool invertZ
)
{
	const int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	const int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if ((x < width) && (y < height) && (x >= 0) && (y >= 0))
	{
		uint nIdx = __umul24(y, width) + x;

		double* pt = pPoints+y*3;
		double* dir = pDirs+y*3;

		float fx = (pt[0] + x * dir[0]) / volumeSize.width;
		float fy = (pt[1] + x * dir[1]) / volumeSize.height;
		float fz = (pt[2] + x * dir[2]) / volumeSize.depth;
		if (!invertZ)
			fz = 1.0 - fz;

		if (fx<0 || fx>1 || fy<0 || fy>1 || fz<0 || fz > 1){
			pData[nIdx] = -2048;
		}
		else{
			pData[nIdx] = 32768*tex3D<float>(volumeTexture, fx, fy, fz);
		}
	}
}

extern "C"
void cu_renderCPR(
	short* pData, 
	int width, 
	int height, 
	cudaTextureObject_t volumeTexture, 
	cudaExtent volumeSize, 
	double* pPoints, 
	double* pDirs, 
	bool invertZ
)
{
	if (NULL == pPoints)
		return;
	short* d_pData = 0;
	checkCudaErrors(cudaMalloc( (void**)&d_pData, width*height*sizeof(short) ));

	double* d_pPoints = 0;
	checkCudaErrors(cudaMalloc( (void**)&d_pPoints, height*3*sizeof(double) ));
	checkCudaErrors(cudaMemcpy( d_pPoints, pPoints, height*3*sizeof(double), cudaMemcpyHostToDevice));

	double* d_pDirs = 0;
	checkCudaErrors(cudaMalloc( (void**)&d_pDirs, height*3*sizeof(double) ));
	checkCudaErrors(cudaMemcpy( d_pDirs, pDirs, height*3*sizeof(double), cudaMemcpyHostToDevice));

	dim3 blockSize(16, 16);
	dim3 gridSize( (width-1)/blockSize.x+1, (height-1)/blockSize.y+1 );
	d_renderCPR<<<gridSize, blockSize>>>(
		d_pData, 
		width, 
		height, 
		volumeTexture, 
		volumeSize, 
		d_pPoints, 
		d_pDirs, 
		invertZ
	);

	checkCudaErrors(cudaMemcpy( pData, d_pData, width*height*sizeof(short), cudaMemcpyDeviceToHost ));

	checkCudaErrors(cudaFree(d_pData));
	checkCudaErrors(cudaFree(d_pPoints));
	checkCudaErrors(cudaFree(d_pDirs));
}


