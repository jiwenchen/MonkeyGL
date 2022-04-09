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

#include "Methods.h"
#include "math.h"
#include <iostream>
#include <cstring>
#include "Defines.h"

using namespace MonkeyGL;

Methods::Methods(void)
{
}

Methods::~Methods(void)
{
}

void Methods::SetSeg( float* m, int n )
{
	for (int i=0; i<n; i++)
	{
		for (int j=0; j<n; j++)
		{
			if (i==j)
				m[i*n+j] = 1;
			else
				m[i*n+j] = 0;
		}
	}
}

void Methods::ComputeTransformMatrix(
    float* pRotateMatrix, 
    float *pTransposRotateMatrix, 
    float* pTransformMatrix, 
    float* pTransposeTransformMatrix, 
    float fxRotate, 
    float fzRotate, 
    float fScale
)
{
	float Rx[9], Rz[9], S[9], Temp[9], RxT[9], RzT[9], ST[9];
	Methods::SetSeg(Rx, 3);
	Methods::SetSeg(Rz, 3);
	Methods::SetSeg(Temp, 3);
	Methods::SetSeg(RxT, 3);
	Methods::SetSeg(RzT, 3);
	Methods::SetSeg(ST, 3);

	S[0] = 1.0f/fScale;	S[1] = 0.0f;		S[2] = 0.0f;
	S[3] = 0.0f;		S[4] = 1.0f/fScale;	S[5] = 0.0f;
	S[6] = 0.0f;		S[7] = 0.0f;		S[8] = 1.0f/fScale;

	Rx[4] = cos((fxRotate)/180.0f*PI);
	Rx[5] = -sin((fxRotate)/180.0f*PI);
	Rx[7] = sin((fxRotate)/180.0f*PI);
	Rx[8] = cos((fxRotate)/180.0f*PI);

	Rz[0] = cos((fzRotate)/180.0f*PI);
	Rz[1] = sin((fzRotate)/180.0f*PI);
	Rz[3] = -sin((fzRotate)/180.0f*PI);
	Rz[4] = cos((fzRotate)/180.0f*PI);

	matrixMul(pTransformMatrix, pTransformMatrix, S,  3, 3, 3);
	matrixMul(pTransformMatrix, pTransformMatrix, Rz, 3, 3, 3);
	matrixMul(pTransformMatrix, pTransformMatrix, Rx, 3, 3, 3);

	Rx[4] = cos((fxRotate)/180.0f*PI);
	Rx[5] = -sin((fxRotate)/180.0f*PI);
	Rx[7] = sin((fxRotate)/180.0f*PI);
	Rx[8] = cos((fxRotate)/180.0f*PI);

	Rz[0] = cos((fzRotate)/180.0f*PI);
	Rz[1] = -sin((fzRotate)/180.0f*PI);
	Rz[3] = sin((fzRotate)/180.0f*PI);
	Rz[4] = cos((fzRotate)/180.0f*PI);

	matrixMul(Temp, Rx, Rz,  3, 3, 3);
	matrixMul(pRotateMatrix, Temp, pRotateMatrix, 3, 3, 3);

	//-------------------------------------------------------
	RxT[4] = cos((fxRotate)/180.0f*PI);
	RxT[5] = sin((fxRotate)/180.0f*PI);
	RxT[7] = -sin((fxRotate)/180.0f*PI);
	RxT[8] = cos((fxRotate)/180.0f*PI);

	RzT[0] = cos((fzRotate)/180.0f*PI);
	RzT[1] = -sin((fzRotate)/180.0f*PI);
	RzT[3] = sin((fzRotate)/180.0f*PI);
	RzT[4] = cos((fzRotate)/180.0f*PI);

	ST[0] = fScale;	ST[1] = 0.0f;	ST[2] = 0.0f;
	ST[3] = 0.0f;	ST[4] = fScale;	ST[5] = 0.0f;
	ST[6] = 0.0f;	ST[7] = 0.0f;	ST[8] = fScale;

	matrixMul(Temp, RxT, RzT,  3, 3, 3);
	matrixMul(Temp, Temp, ST, 3, 3, 3);
	matrixMul(pTransposeTransformMatrix, Temp, pTransposeTransformMatrix, 3, 3, 3);

	RxT[4] = cos((fxRotate)/180.0f*PI);
	RxT[5] = -sin((fxRotate)/180.0f*PI);
	RxT[7] = sin((fxRotate)/180.0f*PI);
	RxT[8] = cos((fxRotate)/180.0f*PI);

	RzT[0] = cos((fzRotate)/180.0f*PI);
	RzT[1] = sin((fzRotate)/180.0f*PI);
	RzT[3] = -sin((fzRotate)/180.0f*PI);
	RzT[4] = cos((fzRotate)/180.0f*PI);

	matrixMul(pTransposRotateMatrix, pTransposRotateMatrix, RzT, 3, 3, 3);
	matrixMul(pTransposRotateMatrix, pTransposRotateMatrix, RxT, 3, 3, 3);
}

void Methods::matrixMul( float *pDst, float *pSrc1, float *pSrc2, int nH1, int nW1, int nW2 )
{
	int i, j, k;

	float *pTemp = new float[nH1*nW2];
	for (i=0; i<nH1; i++)
	{
		for (j=0; j<nW2; j++)
		{
			pTemp[i*nW2+j] = pSrc1[i*nW1]*pSrc2[j];
			for (k=1; k<nW1; k++)
			{
				pTemp[i*nW2+j] += pSrc1[i*nW1+k]*pSrc2[k*nW2+j];
			}
		}
	}
	memcpy(pDst, pTemp, nH1*nW2*sizeof(float));

	delete [] pTemp;
}

Point3d Methods::matrixMul( float *fMatrix, Point3d pt )
{
	double x = fMatrix[0]*pt.x() + fMatrix[1]*pt.y() + fMatrix[2]*pt.z();
	double y = fMatrix[3]*pt.x() + fMatrix[4]*pt.y() + fMatrix[5]*pt.z();
	double z = fMatrix[6]*pt.x() + fMatrix[7]*pt.y() + fMatrix[8]*pt.z();
	return Point3d(x, y, z);
}
