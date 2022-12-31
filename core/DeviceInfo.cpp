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

#include "DeviceInfo.h"
#include <cuda_runtime.h>

using namespace MonkeyGL;

DeviceInfo::DeviceInfo()
{
	m_bInit = false;
	m_nCount = 0;
	m_vecProp.clear();

	cudaError_t error_id = cudaGetDeviceCount(&m_nCount);
	if (error_id != cudaSuccess)
		return;
	if (m_nCount <= 0)
		return;

	int maxMem = -1;
	int maxIdx = -1;
	for (int dev = 0; dev < m_nCount; ++dev)
	{
		if (cudaSuccess != cudaSetDevice(dev))
			continue;
		cudaDeviceProp cuDeviceProp;
		if (cudaSuccess != cudaGetDeviceProperties(&cuDeviceProp, dev))
			continue;
		
		DeviceProp deviceProp;
		memcpy(deviceProp.name, cuDeviceProp.name, sizeof(cuDeviceProp.name));
		deviceProp.totalMem = cuDeviceProp.totalGlobalMem;
		deviceProp.major = cuDeviceProp.major;
		deviceProp.minor = cuDeviceProp.minor;
		m_vecProp.push_back(deviceProp);

		if (deviceProp.totalMem > maxMem){
			maxMem = deviceProp.totalMem;
			maxIdx = dev;
		}
	}

	if (m_vecProp.size() <= 0)
		return;

	if (maxIdx >= 0){
		cudaSetDevice(maxIdx);
	}
	m_bInit = true;
}

DeviceInfo::~DeviceInfo()
{
}

DeviceInfo* DeviceInfo::Instance()
{
	static DeviceInfo* pDeviceInfo = NULL;
	if (NULL == pDeviceInfo)
	{
		pDeviceInfo = new DeviceInfo();
	}
	return pDeviceInfo;
}

bool DeviceInfo::Initialized()
{
	return m_bInit;
}

bool DeviceInfo::GetCount( int& count )
{
	if (!m_bInit)
		return false;
	count = m_nCount;
	return true;
}

bool DeviceInfo::GetName( std::string& strName, const int& index )
{
	if (!m_bInit)
		return false;
	if (index<0 || index>=m_nCount)
		return false;
	strName = m_vecProp[index].name;
	return true;
}

bool DeviceInfo::GetTotalGlobal( unsigned long& mem, const int& index )
{
	if (!m_bInit)
		return false;
	if (index<0 || index>=m_nCount)
		return false;
	mem = m_vecProp[index].totalMem;
	return true;
}

bool DeviceInfo::GetMajor( int& major, const int& index )
{
	if (!m_bInit)
		return false;
	if (index<0 || index>=m_nCount)
		return false;
	major = m_vecProp[index].major;
	return true;
}

bool DeviceInfo::GetMinor( int& minor, const int& index )
{
	if (!m_bInit)
		return false;
	if (index<0 || index>=m_nCount)
		return false;
	minor = m_vecProp[index].minor;
	return true;
}

bool DeviceInfo::SetDevice( const int& index )
{
	return (cudaSuccess == cudaSetDevice(index));
}
