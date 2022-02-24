#include "DeviceInfo.h"
#include <cuda_runtime.h>

using namespace MonkeyGL;

DeviceInfo::DeviceInfo()
{
	bool m_bInit = false;
	m_nCount = 0;
	m_vecProp.clear();

	cudaError_t error_id = cudaGetDeviceCount(&m_nCount);
	if (error_id != cudaSuccess)
		return;
	if (m_nCount <= 0)
		return;

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
	}

	if (m_vecProp.size() <= 0)
		return;

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
