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

#include "TestOnly.h"
#include <driver_types.h>
#include "vector_types.h"

using namespace MonkeyGL;

extern "C" void cu_test_3d(short *h_volumeData, cudaExtent volumeSize);
extern "C" void cu_init_id();
extern "C" bool cu_set_1d(float *h_volumeData, int nLen, unsigned char nLabel);
extern "C" void cu_test_1d(int nLen, unsigned char nLabel);

TestOnly::TestOnly()
{
	testcuda();
}

TestOnly::~TestOnly()
{
}

void TestOnly::testcuda()
{
#if 1
	int nWidth = 512;
	int nHeight = 512;
	int nDepth = 200;
	short *pData = new short[nWidth * nHeight * nDepth];
	for (int i = 0; i < nDepth; i++)
	{
		for (int j = 0; j < nWidth * nHeight; j++)
		{
			pData[i * nHeight * nWidth + j] = j + i;
		}
	}
	cudaExtent volumeSize;
	volumeSize.width = nWidth;
	volumeSize.height = nHeight;
	volumeSize.depth = nDepth;

	cu_test_3d(pData, volumeSize);

	delete[] pData;
#else
	cu_init_id();
	int nLen = 100;
	float *pData = new float[nLen * 4];
	for (int i = 0; i < nLen; i++)
	{
		pData[4 * i] = i;
		pData[4 * i + 1] = i;
		pData[4 * i + 2] = i;
		pData[4 * i + 3] = i + 1;
	}
	cu_set_1d(pData, nLen, 1);

	for (int i = 0; i < nLen; i++)
	{
		pData[4 * i] = i + 20;
		pData[4 * i + 1] = i + 20;
		pData[4 * i + 2] = i + 20;
		pData[4 * i + 3] = i + 20 + 1;
	}
	cu_set_1d(pData, nLen, 10);

	for (int l = 0; l < 10000; l++)
	{
		for (int i = 0; i < 12; i++)
			cu_test_1d(nLen, i);
	}

	delete[] pData;
#endif
}