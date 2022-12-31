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

#include "StopWatch.h"
#include "Logger.h"

#if defined(WIN64) || defined(WIN32) 
#include <time.h>
#include <stdarg.h>
#include <chrono>
#else
#include <sys/time.h>
#include "log4cplus/log4cplus.h"
#endif

using namespace MonkeyGL;

#if defined(WIN64) || defined(WIN32) 

struct timeval
{
	__int64 tv_sec;
	__int64 tv_usec;
};

struct timezone {
	int tz_minuteswest;
	int tz_dsttime;
};

void gettimeofday(struct timeval* tv, struct timezone* tz)
{
	auto time_now = std::chrono::system_clock::now();
	std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
	auto duration_in_s = std::chrono::duration_cast<std::chrono::seconds>(time_now.time_since_epoch()).count();
	auto duration_in_us = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch()).count();
	tv->tv_sec = duration_in_s;
	tv->tv_usec = duration_in_us;
};
#endif

StopWatch::StopWatch(const char * format, ...)
{
    m_starttime_ms = StopWatch::GetMSStamp();
    char buf[4096];
	va_list list;
	va_start(list, format);
	vsnprintf(buf, 4096, format, list);
	va_end(list);
    m_strMsg = std::string(buf);
    Logger::Info("%s begin...", m_strMsg.c_str());
}

StopWatch::~StopWatch()
{
    Logger::Info("%s end. cost %d ms", m_strMsg.c_str(), StopWatch::GetMSStamp() - m_starttime_ms);
}

long long StopWatch::GetMSStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000 + tv.tv_usec/1000);
}