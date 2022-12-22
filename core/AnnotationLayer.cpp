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

#include "BaseLayer.h"
#include "AnnotationLayer.h"
#include "StopWatch.h"
#include "AnnotationUtils.h"

using namespace MonkeyGL;

AnnotationLayer::AnnotationLayer()
{
    m_layerType = LayerTypeAnnotation;
}

AnnotationLayer::~AnnotationLayer()
{
}

bool AnnotationLayer::GetRGBData(std::shared_ptr<unsigned char>& pData, int& nWidth, int& nHeight, PlaneType planeType)
{
    StopWatch sw("AnnotationLayer::GetRGBData");
    if (PlaneVR != planeType){
        return false;
    }
    if (!pData || nWidth <= 10 || nHeight <= 10){
        return false;
    }

    AnnotationUtils::SetFontSize(FontSizeBig);
    AnnotationUtils::Textout2Image("MonkeyGL_005461325464211<=", 10, 200, pData.get(), nWidth, nHeight);

    AnnotationUtils::SetFontSize(FontSizeMiddle);
    int w, h;
    AnnotationUtils::GetSize("Sanasdfa;ajgsdfQian~30<*?", w, h);
    AnnotationUtils::Textout2Image("Sanasdfa;ajgsdfQian~30<*?", 512-w-5, 280, pData.get(), nWidth, nHeight);

    AnnotationUtils::SetFontSize(FontSizeSmall);
    AnnotationUtils::Textout2Image("Hello*WorldZz", 10, 500, pData.get(), nWidth, nHeight);

    return true;
}

bool AnnotationLayer::GetGrayscaleData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, PlaneType planeType)
{
    return true;
}