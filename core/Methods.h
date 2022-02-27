#pragma once
#include "Point.h"

namespace MonkeyGL {

    class Methods
    {
    public:
        Methods(void);
        ~Methods(void);

    public:
        static void SetSeg(float* m, int n);

        static void matrixMul(
            float *pDst, 
            float *pSrc1, 
            float *pSrc2, 
            int nH1, 
            int nW1, 
            int nW2
        );
        static Point3d matrixMul(float *fMatrix, Point3d pt);

        static void ComputeTransformMatrix(
            float* pRotateMatrix,
            float *pTransposRotateMatrix, 
            float* pTransformMatrix, 
            float* pTransposeTransformMatrix, 
            float fxRotate, 
            float fzRotate, 
            float fScale
        );
    };

}
