#include <iostream>

int add(int a, int b){
    int c = a+b;
    std::cout << c << std::endl;
    return c;
}


#include <pybind11/pybind11.h>

PYBIND11_MODULE(example, m) {
    m.def("add", &add);
}