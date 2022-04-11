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
#include "Direction.h"
#include <memory>
#include <string>
#include "Defines.h"

namespace MonkeyGL {

    class VolumeInfo
    {
    public:
        VolumeInfo(void);
        ~VolumeInfo(void);

    public:
        void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        bool LoadVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        int GetVolumeSize(){
            return m_Dims[0]*m_Dims[1]*m_Dims[2];
        }
        int GetVolumeBytes(){
            return GetVolumeSize()*sizeof(short);
        }
        void SetAnisotropy(double x, double y, double z){
            m_Anisotropy[0] = x;
            m_Anisotropy[1] = y;
            m_Anisotropy[2] = z;
        }
        void SetSliceThickness(double sliceTh){
            m_fSliceThickness = sliceTh;
        }

        bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);

        std::shared_ptr<short> GetVolumeData(){
            return m_pVolume;
        }

        std::shared_ptr<short> GetVolumeData(int& nWidth, int& nHeight, int& nDepth){
            nWidth = m_Dims[0];
            nHeight = m_Dims[1];
            nDepth = m_Dims[2];
            return m_pVolume;
        }

        int GetDim(int index){
            return m_Dims[index];
        }
        double GetAnisotropy(int index){
            return m_Anisotropy[index];
        }
        double GetMinAnisotropy()
        {
            double ani = m_Anisotropy[0]<m_Anisotropy[1] ? m_Anisotropy[0] : m_Anisotropy[1];
            return ani<m_Anisotropy[2] ? ani : m_Anisotropy[2];
        }

        bool GetPlaneInitSize(int& nWidth, int& nHeight, int& nNumber, const PlaneType& planeType);

        bool Need2InvertZ();
        bool IsPerpendicularCoord();
        void NormVolumeData();

    private:
        std::shared_ptr<short> m_pVolume;
        double m_fSliceThickness; //mm
        double m_fSlope;
        double m_fIntercept;
        int m_Dims[3];
        double m_Anisotropy[3];
        Point3d m_ptStart;
        Direction3d m_dirX;
        Direction3d m_dirY;
        Direction3d m_dirZ;
    };

}