// MIT License

// Copyright (c) 2022-2023 jiwenchen(cjwbeyond@hotmail.com)

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
#include <string>
#include <map>
#include "Defines.h"
#include <memory>

namespace MonkeyGL{

    class TransferFunction
    {
    public:
        TransferFunction(void);
        ~TransferFunction(void);

    public:
        void SetControlPoints(std::map<int, RGBA> ctrlPts){
            m_pos2rgba = ctrlPts;
            m_pos2alpha.clear();
        }
        void SetControlPoints(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts){
            m_pos2rgba = rgbPts;
            m_pos2alpha = alphaPts;
        }

        void AddControlPoint(int pos, RGBA clr){
            m_pos2rgba[pos] = clr;
        }
        std::map<int, RGBA> GetControlPoints(){
            return m_pos2rgba;
        }

        bool GetTransferFunction(std::shared_ptr<RGBA>& pBuffer, int& nLen);

    private:
        std::map<int, RGBA> m_pos2rgba;
        std::map<int, float> m_pos2alpha;
        int m_nMinPos;
        int m_nMaxPos;
    };

}


