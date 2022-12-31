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

#include "Point.h"
#include "Direction.h"

namespace MonkeyGL{

    class ImageReader
    {
    private:
        /* data */
    public:
        ImageReader(/* args */);
        ~ImageReader();

    public:
        static bool Read(
            const char* szFile,
            std::shared_ptr<short>& pData,
            int dim[],
            double spacing[],
            Point3d& ptOrigin,
            Direction3d& dirX,
            Direction3d& dirY,
            Direction3d& dirZ
        );
        static bool ReadMask(
            const char* szFile,
            std::shared_ptr<unsigned char>& pData,
            int& nWidth,
            int& nHeight,
            int& nDepth
        );
    };
    
}