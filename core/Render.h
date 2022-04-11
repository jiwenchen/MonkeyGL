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

namespace MonkeyGL {

    class Render : public IRender
    {
    public:
        Render(void);
        ~Render(void);

    public:
    // volume info
        virtual bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);
        virtual void SetVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        virtual void SetAnisotropy(double x, double y, double z);

        virtual void SetTransferFunc(const std::map<int, RGBA>& ctrlPoints);
        virtual void SetTransferFunc(const std::map<int, RGBA>& rgbPoints, const std::map<int, double>& alphaPoints);

    // output
        virtual bool GetPlaneMaxSize(int& nWidth, int& nHeight, const PlaneType& planeType);
        virtual bool GetPlaneData(short* pData, int& nWidth, int& nHeight, const PlaneType& planeType);

        virtual bool GetCrossHairPoint(double& x, double& y, const PlaneType& planeType);
        virtual void PanCrossHair(int nx, int ny, PlaneType planeType);

        virtual bool GetVRData(unsigned char* pVR, int nWidth, int nHeight);

        virtual bool GetBatchData( std::vector<short*>& vecBatchData, BatchInfo batchInfo );

        virtual bool GetPlaneRotateMatrix(float* pMatirx, PlaneType planeType);


        virtual void Anterior();
        virtual void Posterior();
        virtual void Left();
        virtual void Right();
        virtual void Head();
        virtual void Foot();

        virtual void Rotate(float fxRotate, float fyRotate);
        virtual void Zoom(float ratio);
        virtual void Pan(float fxShift, float fyShift);
        virtual void SetVRWWWL(float fWW, float fWL);

    private:
        void InitLights();
        void CopyTransferFunc2Device();
        void NormalizeVOI();

        void testcuda();

    private:
        float m_fVOI_xStart;
        float m_fVOI_xEnd;
        float m_fVOI_yStart;
        float m_fVOI_yEnd;
        float m_fVOI_zStart;
        float m_fVOI_zEnd;
        VOI m_voi_Normalize;

        float m_fTotalXTranslate;
        float m_fTotalYTranslate;
        float m_fTotalScale;
        float m_fWW;
        float m_fWL;

        float* m_pRotateMatrix;
        float* m_pTransposRotateMatrix;
        float* m_pTransformMatrix;
        float* m_pTransposeTransformMatrix;
    };
}


