#pragma once
#include <string>
#include <vector>
#include <cstring>

namespace MonkeyGL{

    struct DeviceProp
    {
        char name[256];
        unsigned long totalMem;
        int major;
        int minor;

        char reserved[1024];

        DeviceProp(){
            memset(this, 0, sizeof(DeviceProp));
        }
    };

    class DeviceInfo
    {
    public:
        DeviceInfo();
        ~DeviceInfo();

        static DeviceInfo* Instance();

    public:
        bool GetCount(int& count);
        bool GetName(std::string& strName, const int& index);
        bool GetTotalGlobal(unsigned long& mem, const int& index);
        bool GetMajor(int& major, const int& index);
        bool GetMinor(int& minor, const int& index);

        bool SetDevice(const int& index);

    private:
        bool m_bInit;
        int m_nCount;
        std::vector<DeviceProp> m_vecProp;
    };

}