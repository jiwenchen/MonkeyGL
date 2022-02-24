#include <iostream>
#include "DeviceInfo.h"
#include "Defines.h"

using namespace MonkeyGL;


int main(){
    std::cout << "Hello, I am example1." << std::endl;

    DeviceInfo info;
    int n = 0;
    if (info.GetCount(n)){
        std::cout << n << " device(s)." << std::endl;
    }
    else{
        std::cout << "failed to get device count." << std::endl;
    }
}