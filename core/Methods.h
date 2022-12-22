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
#include "Point.h"
#include "Defines.h"
#include "Direction.h"

namespace MonkeyGL {

    class Methods
    {
    public:
        Methods(void);
        ~Methods(void);

    public:
        static void SetIdentityMatrix(float* m, int n);

        static void MatrixMul(
            float *pDst, 
            float *pSrc1, 
            float *pSrc2, 
            int nH1, 
            int nW1, 
            int nW2
        );
        static Point3d MatrixMul(float *fMatrix, Point3d pt);

        static void ComputeTransformMatrix(
            float* pRotateMatrix,
            float *pTransposRotateMatrix, 
            float* pTransformMatrix, 
            float* pTransposeTransformMatrix, 
            float fxRotate, 
            float fzRotate, 
            float fScale
        );

        static void DrawDotInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, int x, int y, RGB clr=RGB(1.0, 1.0, 1.0));
        static void DrawDotInImage8Bit(unsigned char* pVR, int nWidth, int nHeight, int x, int y, unsigned char brightness=255);
        static void DrawLineInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, float x0, float y0, float x1, float y1, int nLineWidth=2, RGB clr=RGB(1.0, 1.0, 1.0));
        static void DrawLineInImage8Bit(unsigned char* pVR, int nWidth, int nHeight, float x0, float y0, float x1, float y1, int nLineWidth=2, unsigned char brightness=255);
        static void DrawCircleInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, float x, float y, float r, int nLineWidth=2, RGB clr=RGB(1.0, 1.0, 1.0));
        static void DrawTriangleInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, Point2f v1, Point2f v2, Point2f v3, int nLineWidth=2, RGB clr=RGB(1.0, 1.0, 1.0));
        static void FillHoleInImage24Bit(unsigned char* pVR, float* pZBuffer, int nWidth, int nHeight, Point2f v1, Point2f v2, Point2f v3, float zBuffer, RGB clr=RGB(1.0, 1.0, 1.0));

        static void FillHoleInImage_Ch1(float* pImage, float* pZBuffer, int nWidth, int nHeight, float diffuese, float zBuffer, Point2f v1, Point2f v2, Point2f v3);

        static double Distance_Point2Line(Point3d pt, Direction3d dir, Point3d ptLine);
        static double Length_VectorInLine(Point3d pt, Direction3d dir, Point3d ptLine);
        static Point3d Projection_Point2Line(Point3d pt, Direction3d dir, Point3d ptLine);
        static double Distance_Point2Plane(Point3d pt, Direction3d dirH, Direction3d dirV, Point3d ptPlane);
        static double Distance_Point2Plane(Point3d pt, Direction3d dirNorm, Point3d ptPlane);
        static Point3d Projection_Point2Plane(Point3d pt, Direction3d dirH, Direction3d dirV, Point3d ptPlane);
        static Point3d Projection_Point2Plane(Point3d pt, Direction3d dirNorm, Point3d ptPlane);
        static Point3d RotatePoint3D(Point3d pt, Direction3d dirNorm, Point3d ptCenter, float theta);
        static int GetLengthofCrossLineInBox(Direction3d dir, double spacing, double xMin, double xMax, double yMin, double yMax, double zMin, double zMax);

        static int TrimValue(int nValue, int nMin, int nMax){
            nValue = nValue>=nMin ? nValue:nMin;
            nValue = nValue<=nMax ? nValue:nMin;
            return nValue;
        }

        static Point3d GetTransferPoint(double m[3][3], Point3d pt);
        static Point3f GetTransferPointf(float m[9], Point3f pt);
    };

}
