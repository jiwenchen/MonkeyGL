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
#include <map>
#include <string>
#include <memory>
#include "Defines.h"

namespace MonkeyGL {

    struct Mask
    {
        Mask(){
            width = -1;
            height = -1;
            pImg.reset();
        }
        Mask(int w, int h, std::shared_ptr<unsigned char> pd){
            width = w;
            height = h;
            pImg = pd;
        }
        int width;
        int height;
        std::shared_ptr<unsigned char> pImg;
    };
    

    class AnnotationUtils
    {
    public:
        AnnotationUtils(/* args */);
        ~AnnotationUtils();

    public:
        static bool Init();
        static void SetFontSize(FontSize fs);
        static bool Textout2Image(std::string str, int x, int y, AnnotationFormat annoFormat, RGBA clr, unsigned char* pImg, int nWidth, int nHeight);
        static void GetSize(std::string str, int& nWidth, int& nHeight);

    private:
        static std::shared_ptr<unsigned char> GetCharImage(std::string b64, int& nWidth, int nHeight);
        static Mask& GetCharMask(std::string c);
        static Mask DownSampling(const Mask& mask, int ratio);

    private:
        static std::map<std::string, std::map<FontSize, Mask> > m_char2Mask;
        static FontSize m_fontSize;
    };
}