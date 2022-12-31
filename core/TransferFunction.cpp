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

#include "TransferFunction.h"

using namespace MonkeyGL;

TransferFunction::TransferFunction(void)
{
	m_nMinPos = 0;
	m_nMaxPos = 99;
}


TransferFunction::~TransferFunction(void)
{
}

bool TransferFunction::GetTransferFunction( std::shared_ptr<RGBA>& pBuffer, int& nLen )
{
	std::map<int, RGBA>::iterator iter = m_pos2rgba.begin();
	while (iter != m_pos2rgba.end())
	{
		if (iter->first<m_nMinPos || iter->first>m_nMaxPos)
		{
			m_pos2rgba.erase(iter);
			iter = m_pos2rgba.begin();
			continue;
		}
		iter++;
	}

	if (m_pos2rgba.size() < 2)
		return false;

	if (m_pos2rgba.begin()->first != m_nMinPos)
	{
		RGBA& rgba = m_pos2rgba.begin()->second;
		m_pos2rgba[m_nMinPos] = RGBA(rgba.red, rgba.green, rgba.blue, rgba.alpha);
	}
	if (m_pos2rgba.rbegin()->first != m_nMaxPos)
	{
		RGBA& rgba = m_pos2rgba.rbegin()->second;
		m_pos2rgba[m_nMaxPos] = RGBA(rgba.red, rgba.green, rgba.blue, rgba.alpha);
	}

	nLen = m_nMaxPos - m_nMinPos + 1;
	pBuffer.reset(new RGBA[nLen]);

	iter = m_pos2rgba.begin();
	int posPrev = iter->first;
	RGBA rgbaPrev = iter->second;
	iter++;
	while (iter != m_pos2rgba.end())
	{
		int posCur = iter->first;
		RGBA rgbaCur = iter->second;
		int nLenPart = posCur - posPrev + 1;

		for (int j=posPrev; j<=posCur; j++)
		{
			int index = j-m_nMinPos;

			pBuffer.get()[index].red = rgbaPrev.red + (rgbaCur.red - rgbaPrev.red)*(j-posPrev)/nLenPart;
			pBuffer.get()[index].green = rgbaPrev.green + (rgbaCur.green - rgbaPrev.green)*(j-posPrev)/nLenPart;
			pBuffer.get()[index].blue = rgbaPrev.blue + (rgbaCur.blue - rgbaPrev.blue)*(j-posPrev)/nLenPart;

			pBuffer.get()[index].alpha = rgbaPrev.alpha + (rgbaCur.alpha - rgbaPrev.alpha)*(j-posPrev)/nLenPart;
		}
		posPrev = iter->first;
		rgbaPrev = iter->second;
		iter++;
	}

	if (m_pos2alpha.size() > 1)
	{
		std::map<int, float>::iterator iterAlpha = m_pos2alpha.begin();
		while (iterAlpha != m_pos2alpha.end())
		{
			if (iterAlpha->first<m_nMinPos || iterAlpha->first>m_nMaxPos)
			{
				m_pos2alpha.erase(iterAlpha);
				iterAlpha = m_pos2alpha.begin();
				continue;
			}
			iterAlpha++;
		}

		if (m_pos2alpha.begin()->first != m_nMinPos)
		{
			m_pos2alpha[m_nMinPos] = 0;
		}
		if (m_pos2alpha.rbegin()->first != m_nMaxPos)
		{
			float& alpha = m_pos2alpha.rbegin()->second;
			m_pos2alpha[m_nMaxPos] = alpha;
		}

		iterAlpha = m_pos2alpha.begin();
		int posPrev = iterAlpha->first;
		float alphaPrev = iterAlpha->second;
		iterAlpha++;
		while (iterAlpha != m_pos2alpha.end())
		{
			int posCur = iterAlpha->first;
			float alphaCur = iterAlpha->second;
			int nLenPart = posCur - posPrev + 1;

			for (int j=posPrev; j<=posCur; j++)
			{
				int index = j-m_nMinPos;

				pBuffer.get()[index].alpha = alphaPrev + (alphaCur- alphaPrev)*(j-posPrev)/nLenPart;
			}
			posPrev = iterAlpha->first;
			alphaPrev= iterAlpha->second;
			iterAlpha++;
		}
	}
	return true;
}
