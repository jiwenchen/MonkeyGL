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
        bool Initialized();
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