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
#include "Direction.h"
#include "Point.h"
#include "Defines.h"

namespace MonkeyGL{

	class PlaneInfo
	{
	public:
		PlaneInfo(void);
		~PlaneInfo(void);

	public:
		Direction3d GetNormDirection(){
			return m_dirH.cross(m_dirV);
		}

		Point3d GetLeftTopPoint(Point3d ptCenter){
			Point3d ptLeftTop = ptCenter - m_dirH*(0.5*m_nWidth*m_fPixelSpacing);
			ptLeftTop = ptLeftTop - m_dirV*(0.5*m_nHeight*m_fPixelSpacing);
			return ptLeftTop;
		}

	public:
		PlaneType m_PlaneType;
		Direction3d m_dirH;
		Direction3d m_dirV;
		int m_nWidth;
		int m_nHeight;
		int m_nNumber;
		double m_fPixelSpacing;
		double m_fSliceThickness;
		MPRType m_MPRType;
	};

}
