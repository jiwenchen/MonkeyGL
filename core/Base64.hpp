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
#include <iostream>

namespace MonkeyGL {

    static std::string Base64Table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static char Base64DecodeTable[] = {
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -2, -2, -1, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 62, -2, -2, -2, 63,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -2, -2, -2, -2, -2, -2,
        -2,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -2, -2, -2, -2, -2,
        -2, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2
    };

    class Base64
    {
    public:

        static std::string Encode(const unsigned char* pSrc, size_t nLen){
            std::string strResult;
            const unsigned char* pCur = pSrc;

            while(nLen > 2) {
                strResult += Base64Table[pCur[0] >> 2];
                strResult += Base64Table[((pCur[0] & 0x03) << 4) + (pCur[1] >> 4)];
                strResult += Base64Table[((pCur[1] & 0x0f) << 2) + (pCur[2] >> 6)];
                strResult += Base64Table[pCur[2] & 0x3f];

                pCur += 3;
                nLen -= 3;
            }

            if(nLen > 0)
            {
                strResult += Base64Table[pCur[0] >> 2];
                if(nLen%3 == 1) {
                    strResult += Base64Table[(pCur[0] & 0x03) << 4];
                    strResult += "==";
                } else if(nLen%3 == 2) {
                    strResult += Base64Table[((pCur[0] & 0x03) << 4) + (pCur[1] >> 4)];
                    strResult += Base64Table[(pCur[1] & 0x0f) << 2];
                    strResult += "=";
                }
            }
            return strResult;

        };

        static std::string Decode(const char* pSrc, unsigned int nLen){
            int bin = 0, i = 0;
            std::string strResult;
            const char* pCur = pSrc;
            char ch;
            while( (ch = *pCur++) != '\0' && nLen-- > 0 )
            {
                if (ch == '=') {
                    if (*pCur != '=' && (i%4) == 1) {
                        return NULL;
                    }
                    continue;
                }
                ch = Base64DecodeTable[ch];
                if (ch < 0 ) {
                    continue;
                }
                switch(i%4)
                {
                    case 0:
                        bin = ch << 2;
                        break;
                    case 1:
                        bin |= ch >> 4;
                        strResult += bin;
                        bin = ( ch & 0x0f ) << 4;
                        break;
                    case 2:
                        bin |= ch >> 2;
                        strResult += bin;
                        bin = ( ch & 0x03 ) << 6;
                        break;
                    case 3:
                        bin |= ch;
                        strResult += bin;
                        break;
                }
                i++;
            }
            return strResult;
        };
    };
}