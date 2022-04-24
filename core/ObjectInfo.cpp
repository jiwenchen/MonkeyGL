#include "ObjectInfo.h"
#include "Logger.h"

using namespace MonkeyGL;

void ObjectInfo::Print()
{
    Logger::Info("ObjectInfo::Print: alpha[%.2f], ww[%.2f], wl[%.2f]", alpha, ww, wl);
}