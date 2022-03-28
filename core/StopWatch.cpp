#include "StopWatch.h"
#include <sys/time.h>

using namespace MonkeyGL;


StopWatch::StopWatch(const char* szFile)
{
    m_starttime_ms = StopWatch::GetMSStamp();
    m_strMsg = std::string(szFile);
    std::cout << m_strMsg << " begin..." << std::endl;
}

StopWatch::~StopWatch()
{
    std::cout << m_strMsg << " end. cost " << StopWatch::GetMSStamp() - m_starttime_ms << " ms" << std::endl;
}

long long StopWatch::GetMSStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000 + tv.tv_usec/1000);
}