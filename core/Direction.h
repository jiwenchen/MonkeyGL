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
#include "Point.h"

namespace MonkeyGL{

    template <unsigned char dim>
    class Direction
    {
    public:
        Direction(void){
            m_Coords[0] = 1;
            m_Coords[1] = 0;
            if (dim == 3)
                m_Coords[2] = 0;
        };
        Direction(double x, double y){
            if (dim < 2)
                return;
            m_Coords[0] = x;
            m_Coords[1] = y;

            Normalize();
        }
        Direction(double x, double y, double z){
            if (dim < 3)
                return;
            m_Coords[0] = x;
            m_Coords[1] = y;
            m_Coords[2] = z;

            Normalize();
        }
        Direction(Point<double, dim> ptStart, Point<double, dim> ptEnd){
            m_Coords[0] = ptEnd.x() - ptStart.x();
            m_Coords[1] = ptEnd.y() - ptStart.y();
            if (dim == 3)
                m_Coords[2] = ptEnd.z() - ptStart.z();

            Normalize();
        }
        ~Direction(void){
        };

    public:
        double x(){
            return m_Coords[0];
        }
        void SetX(double val){
            m_Coords[0] = val;
        }
        double y(){
            return m_Coords[1];
        }
        void SetY(double val){
            m_Coords[1] = val;
        }
        double z(){
            if (dim < 3)
                return 0;
            return m_Coords[2];
        }
        void SetZ(double val){
            if(dim < 3)
                return;
            m_Coords[2] = val;
        }

        Point<double, dim> operator*(double fr){
            Point<double, dim> ptOutput;
            ptOutput.SetX(x()*fr);
            ptOutput.SetY(y()*fr);
            if (dim == 3)
                ptOutput.SetZ(z()*fr);
            return ptOutput;
        }

        Direction<dim> cross(Direction<dim> dir){
            Direction<dim> dirOutput;
            if(dim == 3)
            {
                double x = this->y()*dir.z() - this->z()*dir.y();
                double y = this->z()*dir.x() - this->x()*dir.z();
                double z = this->x()*dir.y() - this->y()*dir.x();
                dirOutput = Direction<dim>(x, y, z);
            }
            return dirOutput;
        }

        Direction<dim> negative(){
            Direction<dim> dirOutput;
            if(dim == 3)
            {
                double x = -this->x();
                double y = -this->y();
                double z = -this->z();
                dirOutput = Direction<dim>(x, y, z);
            }
            return dirOutput;
        }

        double dot(Direction<dim> dir){
            double v = 0;
            v += this->x() * dir.x();
            v += this->y() * dir.y();
            if(dim == 3)
            {
                v += this->z() * dir.z();
            }
            return v;
        }

        double Length()
        {
            if (dim == 2)
                return sqrt(m_Coords[0]*m_Coords[0]+m_Coords[1]*m_Coords[1]);
            else if (dim == 3)
                return sqrt(m_Coords[0]*m_Coords[0]+m_Coords[1]*m_Coords[1]+m_Coords[2]*m_Coords[2]);
            return 1;
        }

    private:
        void Normalize()
        {
            if (dim < 2)
                return;

            double n = Length();
            m_Coords[0] /= n;
            m_Coords[1] /= n;
            if (dim == 3)
                m_Coords[2] /= n;
        }

    private:
        double m_Coords[dim];
    };

    typedef Direction<3> Direction3d;
    typedef Direction<2> Direction2d;

}

