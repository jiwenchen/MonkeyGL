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
        bool LoadVolumeFile(const char* szFile);
        bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);
        bool AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel);
        bool UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel);

        std::shared_ptr<short> GetVolumeData(){
            return m_pVolume;
        }

        std::shared_ptr<short> GetVolumeData(int& nWidth, int& nHeight, int& nDepth){
            nWidth = m_Dims[0];
            nHeight = m_Dims[1];
            nDepth = m_Dims[2];
            return m_pVolume;
        }

        std::shared_ptr<unsigned char> GetMaskData(){
            return m_pMask;
        }

        void Clear();

        void SetDirection(Direction3d dirX, Direction3d dirY, Direction3d dirZ);
        int GetVolumeSize(){
            return m_Dims[0]*m_Dims[1]*m_Dims[2];
        }
        int GetVolumeBytes(){
            return GetVolumeSize()*sizeof(short);
        }
        void SetSpacing(double x, double y, double z);

        void SetOrigin(Point3d pt){
            m_ptOriginPatient = pt;
        }

        Point3d GetOrigin(){
            return m_ptOriginPatient;
        }

        Point3d Voxel2Patient(Point3d ptVoxel);
        Point3d Patient2Voxel(Point3d ptPatient);

        bool HasVolumeData(){
            return bool(m_pVolume);
        }

        int GetDim(int index){
            return m_Dims[index];
        }
        double GetSpacing(int index){
            return m_Spacing[index];
        }
        double GetMinSpacing()
        {
            double ani = m_Spacing[0]<m_Spacing[1] ? m_Spacing[0] : m_Spacing[1];
            return ani<m_Spacing[2] ? ani : m_Spacing[2];
        }

        bool Need2InvertZ();
        bool IsPerpendicularCoord();
        void NormVolumeData();
        std::shared_ptr<unsigned char> CheckAndNormMaskData(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth);
        std::shared_ptr<unsigned char> NormMaskData(std::shared_ptr<unsigned char>pData);

    private:
        std::shared_ptr<short> m_pVolume;
        bool m_bVolumeHasInverted;
        std::shared_ptr<unsigned char> m_pMask;
        int m_Dims[3];
        double m_Spacing[3];
        Point3d m_ptOriginPatient;
        Direction3d m_dirX;
        Direction3d m_dirY;
        Direction3d m_dirZ;
    };

}