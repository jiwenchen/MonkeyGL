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
#include "Defines.h"
#include "TransferFunction.h"
#include "VolumeInfo.h"
#include "Methods.h"
#include "IRender.h"
#include "VRProvider.h"
#include "MPRProvider.h"
#include "CPRProvider.h"

namespace MonkeyGL {

    class Render : public IRender
    {
    public:
        Render(void);
        ~Render(void);

    public:

    // output
        virtual bool GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);

        virtual bool GetCrossHairPoint(double& x, double& y, const PlaneType& planeType);
        virtual void PanCrossHair(float fx, float fy, PlaneType planeType);

        virtual bool GetVRData(std::shared_ptr<unsigned char>& pData, int nWidth, int nHeight);

        virtual bool GetBatchData( std::vector<short*>& vecBatchData, BatchInfo batchInfo );

        virtual void ShowPlaneInVR(bool bShow);

    private:
        bool GetMPRPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);
        bool GetCPRPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);

    private:
        VRProvider m_vrProvider;
        MPRProvider m_mprProvider;
        CPRProvider m_cprProvider;
    };
}


