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

namespace MonkeyGL{

    class RenderInfo
    {
    public:
        RenderInfo(/* args */);
        ~RenderInfo();

    public:
        bool SetVRSize(int nWidth, int nHeight);
        void GetVRSize(int& nWidth, int& nHeight);

        void Rotate(float fxRotate, float fyRotate);
        float Zoom(float ratio);
        float GetZoomRatio();
        void Pan(float fxShift, float fyShift);

        void Anterior();
        void Posterior();
        void Left();
        void Right();
        void Head();
        void Foot();

        float GetTotalXTranslate(){
            return m_fTotalXTranslate;
        }

        float GetTotalYTranslate(){
            return m_fTotalYTranslate;
        }

        float GetTotalScale(){
            return m_fTotalScale;
        }

        float* GetRotateMatrix(){
            return m_pRotateMatrix;
        }

    private:
        void Init();

    private:
        friend class DataManager;
        
        int m_nWidth_VR;
        int m_nHeight_VR;

        float m_fTotalXTranslate;
        float m_fTotalYTranslate;
        float m_fTotalScale;

        float m_pRotateMatrix[9];
        float m_pTransposRotateMatrix[9];
        float m_pTransformMatrix[9];
        float m_pTransposeTransformMatrix[9];
    };
}