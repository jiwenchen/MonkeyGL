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
#include "CPRLineLayer.h"
#include "StopWatch.h"
#include "DataManager.h"
#include "Methods.h"

using namespace MonkeyGL;

CPRLineLayer::CPRLineLayer()
{
    m_layerType = LayerTypeCPRLine;
}

CPRLineLayer::~CPRLineLayer()
{
}

bool CPRLineLayer::GetRGBData(std::shared_ptr<unsigned char>& pData, int& nWidth, int& nHeight, PlaneType planeType)
{
    StopWatch sw("CPRLineLayer::GetRGBData");
	if (!DataManager::Instance()->IsLayerEnable(GetLayerType())){
		return false;
	}

    if (PlaneVR != planeType){
        return false;
    }

    std::vector<Point3d> cprLine = DataManager::Instance()->GetCPRLineVoxel();
    RGBA clr = DataManager::Instance()->GetCPRInfo().GetLineColor();
    if (cprLine.size() >= 2){
        float x0, y0, x1, y1;
        DataManager::Instance()->TransferVoxel2ImageInVR(x0, y0, nWidth, nHeight, cprLine[0]);
        for (int i=1; i<cprLine.size(); i++){
            DataManager::Instance()->TransferVoxel2ImageInVR(x1, y1, nWidth, nHeight, cprLine[i]);
            Methods::DrawLineInImage24Bit(pData.get(), nWidth, nHeight, x0, y0, x1, y1, 1, clr);
            x0 = x1;
            y0 = y1;
        }
    }

    return true;
}

bool CPRLineLayer::GetGrayscaleData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, PlaneType planeType)
{
	if (!DataManager::Instance()->IsLayerEnable(GetLayerType())){
		return false;
	}
    return false;
}