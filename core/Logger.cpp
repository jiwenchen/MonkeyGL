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

#include "Logger.h"
#include "log4cplus/log4cplus.h"

using namespace MonkeyGL;

log4cplus::Initializer logInitializer;
static log4cplus::SharedAppenderPtr consoleAppender(new log4cplus::ConsoleAppender);
static log4cplus::Logger logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT ("MonkeyGL"));


extern "C"
void initLogger(){
    consoleAppender->setName(LOG4CPLUS_TEXT("console"));

    consoleAppender->setLayout(std::unique_ptr<log4cplus::Layout>(new log4cplus::SimpleLayout()));
    log4cplus::tstring pattern = LOG4CPLUS_TEXT("%D{%y-%m-%d %H:%M:%S.%q} [%t] %p: %m%n");
    consoleAppender->setLayout(std::unique_ptr<log4cplus::Layout>(new log4cplus::PatternLayout(pattern)));

    logger.setLogLevel(log4cplus::INFO_LOG_LEVEL);
    logger.addAppender(consoleAppender);
}

std::string Logger::FormatMsg(const char * format, ...)
{
	char buf[4096];
	va_list list;
	va_start(list, format);
	vsnprintf(buf, 4096, format, list);
	va_end(list);
	return std::string(buf);
}

void Logger::Init(){
    initLogger();
}

void Logger::SetLevel(LogLevel level){
    switch (level)
    {
    case LogLevelInfo:
        logger.setLogLevel(log4cplus::INFO_LOG_LEVEL);
        break;
    case LogLevelWarn:
        logger.setLogLevel(log4cplus::WARN_LOG_LEVEL);
        break;
    case LogLevelError:
        logger.setLogLevel(log4cplus::ERROR_LOG_LEVEL);
        break;
    
    default:
        break;
    }
}

void Logger::Info(const char * format, ...){
	char buf[4096];
	va_list list;
	va_start(list, format);
	vsnprintf(buf, 4096, format, list);
	va_end(list);
    LOG4CPLUS_INFO(logger, LOG4CPLUS_TEXT(buf));
}

void Logger::Warn(const char * format, ...){
	char buf[4096];
	va_list list;
	va_start(list, format);
	vsnprintf(buf, 4096, format, list);
	va_end(list);
    LOG4CPLUS_WARN(logger, LOG4CPLUS_TEXT(buf));
}

void Logger::Error(const char * format, ...){
	char buf[4096];
	va_list list;
	va_start(list, format);
	vsnprintf(buf, 4096, format, list);
	va_end(list);
    LOG4CPLUS_ERROR(logger, LOG4CPLUS_TEXT(buf));
}