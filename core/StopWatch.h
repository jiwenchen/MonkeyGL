#pragma once
#include "time.h"
#include <iostream>

namespace MonkeyGL {

    class StopWatch
    {
    public:
        StopWatch(const char* sMsg);
        ~StopWatch();

    static long long GetMSStamp();

    private:
        long long m_starttime_ms;
        std::string m_strMsg;
    };
}