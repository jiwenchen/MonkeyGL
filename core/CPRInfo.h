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
#include <vector>
#include <memory>
#include "Point.h"
#include "Direction.h"
#include "Defines.h"

namespace MonkeyGL {
    class DataManager;

    class CPRInfo
    {
    public:
        CPRInfo();
        ~CPRInfo();

    public:
        void SetDataManager(DataManager* pDataManager);
        void SetSpacing(Point3d spacing);
        bool SetCPRLinePatient(std::vector<Point3d> cprLine);
        bool SetCPRLineVoxel(std::vector<Point3d> cprLine);
        std::vector<Point3d> GetCPRLineVoxel();
        bool RotateCPR(float angle, PlaneType planeType);
        bool GetCPRInfo(Point3d*& pPoints, Direction3d*& pDirs, int& nWidth, int &nHeight, PlaneType planeType);

    private:
        bool UpdateCPRInfo(); 
        float GetStretchedRadius(){
            return m_StretchedRadius;
        }
        float GetStraightenedRadius(){
            return m_StraightenedRadius;
        }

        bool GetCPRInfoStretched(Point3d*& pPoints, Direction3d*& pDirs, int& nWidth, int& nHeight);
        bool GetCPRInfoStraightened(Point3d*& pPoints, Direction3d*& pDirs, int& nWidth, int& nHeight);

        static Direction3d FirstDirectionProjection(Point3d pt, Direction3d dirN);

    private:
        double m_StretchedRadius;
        double m_StraightenedRadius;
        std::vector<Point3d> m_cprLineVoxel;
        Direction3d m_dirStretchedCPR;
        Point3d m_ptOriginStrechedCPR;
        Point3d m_ptOriginStraightenedCPR;
        double m_angleStrechedCPR;
        double m_angleStraightenedCPR;
        Point3d m_spacing;
        double m_minSpacing;
        DataManager* m_pDataManager;
    };
}