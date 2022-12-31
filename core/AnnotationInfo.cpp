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

#include "AnnotationInfo.h"
#include "Methods.h"
#include "DataManager.h"
#include "Logger.h"

using namespace MonkeyGL;

AnnotationInfo::AnnotationInfo()
{
    m_annotations.clear();
}

AnnotationInfo::~AnnotationInfo()
{
    m_annotations.clear();
}


bool AnnotationInfo::AddAnnotation(PlaneType planeType, std::string txt, int x, int y, FontSize fontSize, AnnotationFormat annoFormat, RGBA clr)
{
    AnnotationDef anno;
    anno.annoFormat = annoFormat;
    anno.fontSize = fontSize;
    anno.x = x;
    anno.y = y;
    anno.strText = txt;
    anno.color = clr;
    if (m_annotations.find(planeType) != m_annotations.end()){
        m_annotations[planeType].push_back(anno);
    }
    else{
        std::vector<AnnotationDef> annos;
        annos.push_back(anno);
        m_annotations[planeType] = annos;
    }

    return true;
}

bool AnnotationInfo::RemovePlaneAnnotations(PlaneType planeType)
{
    std::map<PlaneType, std::vector<AnnotationDef> >::iterator iter = m_annotations.find(planeType);
    if (iter != m_annotations.end()){
        m_annotations.erase(iter);
    }
    return true;
}

bool AnnotationInfo::RemoveAllAnnotations()
{
    m_annotations.clear();
    return true;
}

std::vector<AnnotationDef> AnnotationInfo::GetAnnotations(PlaneType planeType)
{
    std::vector<AnnotationDef> annotations;
    if (m_annotations.find(planeType) != m_annotations.end()){
        annotations = m_annotations[planeType];
    }
    return annotations;
}
