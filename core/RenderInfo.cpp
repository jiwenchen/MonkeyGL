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

#include "RenderInfo.h"
#include "Methods.h"

using namespace MonkeyGL;

RenderInfo::RenderInfo()
{
	m_nWidth_VR = 512;
	m_nHeight_VR = 512;

	Init();
}

RenderInfo::~RenderInfo()
{
}

void RenderInfo::Init()
{
	m_fTotalXTranslate = 0.0f;
	m_fTotalYTranslate = 0.0f;
	m_fTotalScale = 1.0f;

	Methods::SetIdentityMatrix(m_pRotateMatrix, 3);
	Methods::SetIdentityMatrix(m_pTransposRotateMatrix, 3);
	Methods::SetIdentityMatrix(m_pTransformMatrix, 3);
	Methods::SetIdentityMatrix(m_pTransposeTransformMatrix, 3);
}

bool RenderInfo::SetVRSize(int nWidth, int nHeight)
{
	float delta = (nWidth*nHeight)/(768*768);
	if (delta > 1){
		m_nWidth_VR = nWidth / delta;
		m_nHeight_VR = nHeight / delta;
	}
	else{
		m_nWidth_VR = nWidth;
		m_nHeight_VR = nHeight;
	}

	if (m_nWidth_VR % 2){
		m_nWidth_VR = m_nWidth_VR + 1;
	}
	
	return true;
}

void RenderInfo::GetVRSize(int& nWidth, int& nHeight)
{
	nWidth = m_nWidth_VR;
	nHeight = m_nHeight_VR;
}

void RenderInfo::Rotate(float fxRotate, float fyRotate)
{
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, fyRotate, fxRotate, 1.0f);
}

float RenderInfo::Zoom(float ratio)
{
	m_fTotalScale *= ratio;
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 0.0f, ratio);
	return m_fTotalScale;
}

float RenderInfo::GetZoomRatio()
{
	return m_fTotalScale;
}

void RenderInfo::Pan(float fxShift, float fyShift)
{
	m_fTotalXTranslate += fxShift;
	m_fTotalYTranslate += fyShift;
}

void RenderInfo::Anterior()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 0.0f, m_fTotalScale);
}

void RenderInfo::Posterior()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 180.0f, m_fTotalScale);
}

void RenderInfo::Left()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, -90.0f, m_fTotalScale);
}

void RenderInfo::Right()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 0.0f, 90.0f, m_fTotalScale);
}

void RenderInfo::Head()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, 90.0f, 180.0f, m_fTotalScale);
}

void RenderInfo::Foot()
{
	Init();
	Methods::ComputeTransformMatrix(m_pRotateMatrix, m_pTransposRotateMatrix, m_pTransformMatrix, m_pTransposeTransformMatrix, -90.0f, 0.0f, m_fTotalScale);
}
