// MIT License

// Copyright (c) 2022 jiwenchen(cjwbeyond@hotmail.com)

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
#include <cstring>
#include "math.h"

namespace MonkeyGL{

    template <class T, unsigned char dim>
    class Point
    {
    public:
        Point(void){
            memset(m_Coords, 0, dim*sizeof(T));
        };
        Point(T x, T y){
            if (dim < 2)
                return;
            m_Coords[0] = x;
            m_Coords[1] = y;
        }
        Point(T x, T y, T z){
            if (dim < 3)
                return;
            m_Coords[0] = x;
            m_Coords[1] = y;
            m_Coords[2] = z;
        }
        ~Point(void){
        };

    public:
        void Set(T x, T y){
            if (dim < 2)
                return;
            SetX(x);
            SetY(y);
        }
        void Set(T x, T y, T z){
            if (dim < 3)
                return;
            SetX(x);
            SetY(y);
            SetZ(z);
        }
        T x(){
            return m_Coords[0];
        }
        void SetX(T val){
            m_Coords[0] = val;
        }
        T y(){
            return m_Coords[1];
        }
        void SetY(T val){
            m_Coords[1] = val;
        }
        T z(){
            if (dim < 3)
                return 0;
            return m_Coords[2];
        }
        void SetZ(T val){
            if(dim < 3)
                return;
            m_Coords[2] = val;
        }

        double DistanceTo(Point<T, dim> pt){
            double ds = (x()-pt.x())*(x()-pt.x()) + 
                (y()-pt.y())*(y()-pt.y());
            if (dim == 3)
                ds += (z()-pt.z())*(z()-pt.z());
            return sqrt(ds);
        }

        Point<T, dim> operator+=(Point<T, dim> pt){
            SetX(x()+pt.x());
            SetY(y()+pt.y());
            if(dim == 3)
                SetZ(z()+pt.z());
            return *this;
        }
        Point<T, dim> operator+(Point<T, dim> pt){
            Point<T, dim> ptOutput;
            ptOutput.SetX(x()+pt.x());
            ptOutput.SetY(y()+pt.y());
            if (dim == 3)
                ptOutput.SetZ(z()+pt.z());
            return ptOutput;
        }

        Point<T, dim> operator-=(Point<T, dim> pt){
            SetX(x()-pt.x());
            SetY(y()-pt.y());
            if (dim == 3)
                SetZ(z()-pt.z());
            return *this;
        }
        Point<T, dim> operator-(Point<T, dim> pt){
            Point<T, dim> ptOutput;
            ptOutput.SetX(x()-pt.x());
            ptOutput.SetY(y()-pt.y());
            if (dim == 3)
                ptOutput.SetZ(z()-pt.z());
            return ptOutput;
        }

        T operator[](int idx){
            if (idx<0 || idx>=dim)
                return 0;
            return m_Coords[idx];
        }

    private:
        T m_Coords[dim];
    };

    typedef Point<double, 3> Point3d;
    typedef Point<double, 2> Point2d;

}

