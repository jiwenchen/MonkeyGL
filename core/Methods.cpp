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

#include "Methods.h"
#include "math.h"
#include <iostream>
#include <cstring>

using namespace MonkeyGL;

Methods::Methods(void)
{
}

Methods::~Methods(void)
{
}

void Methods::SetIdentityMatrix( float* m, int n )
{
	for (int i=0; i<n; i++)
	{
		for (int j=0; j<n; j++)
		{
			if (i==j)
				m[i*n+j] = 1;
			else
				m[i*n+j] = 0;
		}
	}
}

void Methods::ComputeTransformMatrix(
    float* pRotateMatrix, 
    float *pTransposRotateMatrix, 
    float* pTransformMatrix, 
    float* pTransposeTransformMatrix, 
    float fxRotate, 
    float fzRotate, 
    float fScale
)
{
	float Rx[9], Rz[9], S[9], Temp[9], RxT[9], RzT[9], ST[9];
	Methods::SetIdentityMatrix(Rx, 3);
	Methods::SetIdentityMatrix(Rz, 3);
	Methods::SetIdentityMatrix(Temp, 3);
	Methods::SetIdentityMatrix(RxT, 3);
	Methods::SetIdentityMatrix(RzT, 3);
	Methods::SetIdentityMatrix(ST, 3);

	S[0] = 1.0f/fScale;	S[1] = 0.0f;		S[2] = 0.0f;
	S[3] = 0.0f;		S[4] = 1.0f/fScale;	S[5] = 0.0f;
	S[6] = 0.0f;		S[7] = 0.0f;		S[8] = 1.0f/fScale;

	Rx[4] = cos((fxRotate)/180.0f*PI);
	Rx[5] = -sin((fxRotate)/180.0f*PI);
	Rx[7] = sin((fxRotate)/180.0f*PI);
	Rx[8] = cos((fxRotate)/180.0f*PI);

	Rz[0] = cos((fzRotate)/180.0f*PI);
	Rz[1] = sin((fzRotate)/180.0f*PI);
	Rz[3] = -sin((fzRotate)/180.0f*PI);
	Rz[4] = cos((fzRotate)/180.0f*PI);

	MatrixMul(pTransformMatrix, pTransformMatrix, S,  3, 3, 3);
	MatrixMul(pTransformMatrix, pTransformMatrix, Rz, 3, 3, 3);
	MatrixMul(pTransformMatrix, pTransformMatrix, Rx, 3, 3, 3);

	Rx[4] = cos((fxRotate)/180.0f*PI);
	Rx[5] = -sin((fxRotate)/180.0f*PI);
	Rx[7] = sin((fxRotate)/180.0f*PI);
	Rx[8] = cos((fxRotate)/180.0f*PI);

	Rz[0] = cos((fzRotate)/180.0f*PI);
	Rz[1] = sin((fzRotate)/180.0f*PI);
	Rz[3] = -sin((fzRotate)/180.0f*PI);
	Rz[4] = cos((fzRotate)/180.0f*PI);

	MatrixMul(Temp, Rx, Rz,  3, 3, 3);
	MatrixMul(pRotateMatrix, Temp, pRotateMatrix, 3, 3, 3);

	//-------------------------------------------------------
	RxT[4] = cos((fxRotate)/180.0f*PI);
	RxT[5] = sin((fxRotate)/180.0f*PI);
	RxT[7] = -sin((fxRotate)/180.0f*PI);
	RxT[8] = cos((fxRotate)/180.0f*PI);

	RzT[0] = cos((fzRotate)/180.0f*PI);
	RzT[1] = -sin((fzRotate)/180.0f*PI);
	RzT[3] = sin((fzRotate)/180.0f*PI);
	RzT[4] = cos((fzRotate)/180.0f*PI);

	ST[0] = fScale;	ST[1] = 0.0f;	ST[2] = 0.0f;
	ST[3] = 0.0f;	ST[4] = fScale;	ST[5] = 0.0f;
	ST[6] = 0.0f;	ST[7] = 0.0f;	ST[8] = fScale;

	MatrixMul(Temp, RxT, RzT,  3, 3, 3);
	MatrixMul(Temp, Temp, ST, 3, 3, 3);
	MatrixMul(pTransposeTransformMatrix, Temp, pTransposeTransformMatrix, 3, 3, 3);

	RxT[4] = cos((fxRotate)/180.0f*PI);
	RxT[5] = sin((fxRotate)/180.0f*PI);
	RxT[7] = -sin((fxRotate)/180.0f*PI);
	RxT[8] = cos((fxRotate)/180.0f*PI);

	RzT[0] = cos((fzRotate)/180.0f*PI);
	RzT[1] = -sin((fzRotate)/180.0f*PI);
	RzT[3] = sin((fzRotate)/180.0f*PI);
	RzT[4] = cos((fzRotate)/180.0f*PI);

	MatrixMul(pTransposRotateMatrix, pTransposRotateMatrix, RzT, 3, 3, 3);
	MatrixMul(pTransposRotateMatrix, pTransposRotateMatrix, RxT, 3, 3, 3);
}

void Methods::MatrixMul( float *pDst, float *pSrc1, float *pSrc2, int nH1, int nW1, int nW2 )
{
	int i, j, k;

	float *pTemp = new float[nH1*nW2];
	for (i=0; i<nH1; i++)
	{
		for (j=0; j<nW2; j++)
		{
			pTemp[i*nW2+j] = pSrc1[i*nW1]*pSrc2[j];
			for (k=1; k<nW1; k++)
			{
				pTemp[i*nW2+j] += pSrc1[i*nW1+k]*pSrc2[k*nW2+j];
			}
		}
	}
	memcpy(pDst, pTemp, nH1*nW2*sizeof(float));

	delete [] pTemp;
}

Point3d Methods::MatrixMul( float *fMatrix, Point3d pt )
{
	double x = fMatrix[0]*pt.x() + fMatrix[1]*pt.y() + fMatrix[2]*pt.z();
	double y = fMatrix[3]*pt.x() + fMatrix[4]*pt.y() + fMatrix[5]*pt.z();
	double z = fMatrix[6]*pt.x() + fMatrix[7]*pt.y() + fMatrix[8]*pt.z();
	return Point3d(x, y, z);
}

void Methods::DrawDotInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, int x, int y, RGBA clr)
{
	if (x<0 || x>=nWidth || y<0 || y>=nHeight){
		return;
	}
	int red = int(clr.red * 255);
	int green = int(clr.green * 255);
	int blue = int(clr.blue * 255);
	pVR[3*(y*nWidth+x)] = red;
	pVR[3*(y*nWidth+x)+1] = green;
	pVR[3*(y*nWidth+x)+2] = blue;
}
void Methods::DrawDotInImage8Bit(unsigned char* pVR, int nWidth, int nHeight, int x, int y, unsigned char brightness)
{
	if (x<0 || x>=nWidth || y<0 || y>=nHeight){
		return;
	}
	pVR[y*nWidth+x] = brightness;
}

void Methods::DrawLineInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, float x0, float y0, float x1, float y1, int nLineWidth, RGBA clr){
	double len = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
	Direction2d dir = Direction2d(x1-x0, y1-y0);
	Direction2d dirN = Direction2d(-dir.y(), dir.x());

	float half = 1.0*nLineWidth/2;
	float x0Fist = x0 - (half-0.5)*dirN.x();
	float y0Fist = y0 - (half-0.5)*dirN.y();
	for (int idx=0; idx<nLineWidth; idx++){
		float x0temp = x0Fist + idx*dirN.x();
		float y0temp = y0Fist + idx*dirN.y();
		for (int i=0; i<=int(len+0.5)*2; i++){
			int x = int(x0temp + i*dir.x()/2.0 + 0.5);
			int y = int(y0temp + i*dir.y()/2.0 + 0.5);
			DrawDotInImage24Bit(pVR, nWidth, nHeight, x, y, clr);
		}
	}
}

void Methods::DrawLineInImage8Bit(unsigned char* pVR, int nWidth, int nHeight, float x0, float y0, float x1, float y1, int nLineWidth, unsigned char brightness){
	double len = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
	Direction2d dir = Direction2d(x1-x0, y1-y0);
	Direction2d dirN = Direction2d(-dir.y(), dir.x());

	float half = 1.0*nLineWidth/2;
	float x0Fist = x0 - (half-0.5)*dirN.x();
	float y0Fist = y0 - (half-0.5)*dirN.y();
	for (int idx=0; idx<nLineWidth; idx++){
		float x0temp = x0Fist + idx*dirN.x();
		float y0temp = y0Fist + idx*dirN.y();
		for (int i=0; i<=int(len+0.5)*2; i++){
			int x = int(x0temp + i*dir.x()/2.0 + 0.5);
			int y = int(y0temp + i*dir.y()/2.0 + 0.5);
			DrawDotInImage8Bit(pVR, nWidth, nHeight, x, y, brightness);
		}
	}
}

void Methods::DrawCircleInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, float x, float y, float r, int nLineWidth, RGBA clr){
	double angDelta = PI/180.0;
	double m[2][2] = {{cos(angDelta), -sin(angDelta)}, {sin(angDelta), cos(angDelta)}};

	float xPre = -r;
	float yPre = 0;
	float xNxt = xPre;
	float yNxt = yPre;
	for (int i=0; i<360; i++){
		xNxt = m[0][0] * xPre + m[0][1] * yPre;
		yNxt = m[1][0] * xPre + m[1][1] * yPre;
		Methods::DrawLineInImage24Bit(pVR, nWidth, nHeight, xPre+x, yPre+y, xNxt+x, yNxt+y, nLineWidth, clr);
		xPre = xNxt;
		yPre = yNxt;
	}
}


void Methods::DrawTriangleInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, Point2f v1, Point2f v2, Point2f v3, int nLineWidth, RGBA clr)
{
	Methods::DrawLineInImage24Bit(pVR, nWidth, nHeight, v1.x(), v1.y(), v2.x(), v2.y(), nLineWidth, clr);
	Methods::DrawLineInImage24Bit(pVR, nWidth, nHeight, v2.x(), v2.y(), v3.x(), v3.y(), nLineWidth, clr);
	Methods::DrawLineInImage24Bit(pVR, nWidth, nHeight, v3.x(), v3.y(), v1.x(), v1.y(), nLineWidth, clr);
}

void Methods::FillHoleInImage24Bit(unsigned char* pVR, float* pZBuffer, int nWidth, int nHeight, Point2f v1, Point2f v2, Point2f v3, float zBuffer, RGBA clr)
{
	float fxmin = min(v1.x(), min(v2.x(), v3.x()));
	float fymin = min(v1.y(), min(v2.y(), v3.y()));
	float fxmax = max(v1.x(), max(v2.x(), v3.x()));
	float fymax = max(v1.y(), max(v2.y(), v3.y()));
	int xmin = int(fxmin);
	int ymin = int(fymin);
	int xmax = int(fxmax+0.5);
	int ymax = int(fymax+0.5);
	if (xmin>=nWidth || ymin>=nHeight || xmax<0 || ymax <0){
		return;
	}

	int nw = xmax - xmin + 1;
	int nh = ymax - ymin + 1;

	std::shared_ptr<unsigned char> pMask(new unsigned char[nw * nh]);
	memset(pMask.get(), 0, nw * nh); 
	Methods::DrawLineInImage8Bit(pMask.get(), nw, nh, v1.x()-xmin, v1.y()-ymin, v2.x()-xmin, v2.y()-ymin, 1);
	Methods::DrawLineInImage8Bit(pMask.get(), nw, nh, v1.x()-xmin, v1.y()-ymin, v3.x()-xmin, v3.y()-ymin, 1);
	Methods::DrawLineInImage8Bit(pMask.get(), nw, nh, v2.x()-xmin, v2.y()-ymin, v3.x()-xmin, v3.y()-ymin, 1);

	for (int y=0; y<nh; y++){
		for (int x=0; x<nw; x++){
			if (pMask.get()[y*nw+x] == 0){
				pMask.get()[y*nw+x] = 1;
			}
			else{
				break;
			}
		}
		for (int x=nw-1; x>=0; x--){
			if (pMask.get()[y*nw+x] == 0){
				pMask.get()[y*nw+x] = 1;
			}
			else{
				break;
			}
		}
	}

	for (int y=0; y<nh; y++){
		int yIdx = y+ymin;
		if (yIdx >= nHeight || yIdx < 0){
			continue;
		}
		for (int x=0; x<nw; x++){
			int xIdx = x + xmin;
			if (xIdx >= nWidth || xIdx < 0){
				continue;
			}
			if (pMask.get()[y*nw+x] != 1){
				if (pZBuffer[yIdx*nWidth+xIdx] <= 0 || pZBuffer[yIdx*nWidth+xIdx] < zBuffer){
					Methods::DrawDotInImage24Bit(pVR, nWidth, nHeight, xIdx, yIdx, clr);
					pZBuffer[yIdx*nWidth+xIdx] = zBuffer;
				}
			}
		}
	}
}

void Methods::FillHoleInImage_Ch1(float* pImage, float* pZBuffer, int nWidth, int nHeight, float diffuese, float zBuffer, Point2f v1, Point2f v2, Point2f v3)
{
	float fxmin = min(v1.x(), min(v2.x(), v3.x()));
	float fymin = min(v1.y(), min(v2.y(), v3.y()));
	float fxmax = max(v1.x(), max(v2.x(), v3.x()));
	float fymax = max(v1.y(), max(v2.y(), v3.y()));
	int xmin = int(fxmin);
	int ymin = int(fymin);
	int xmax = int(fxmax+0.5);
	int ymax = int(fymax+0.5);
	if (xmin>=nWidth || ymin>=nHeight || xmax<0 || ymax <0){
		return;
	}

	int nw = xmax - xmin + 1;
	int nh = ymax - ymin + 1;

	std::shared_ptr<unsigned char> pMask(new unsigned char[nw * nh]);
	memset(pMask.get(), 0, nw * nh); 
	Methods::DrawLineInImage8Bit(pMask.get(), nw, nh, v1.x()-xmin, v1.y()-ymin, v2.x()-xmin, v2.y()-ymin, 1);
	Methods::DrawLineInImage8Bit(pMask.get(), nw, nh, v1.x()-xmin, v1.y()-ymin, v3.x()-xmin, v3.y()-ymin, 1);
	Methods::DrawLineInImage8Bit(pMask.get(), nw, nh, v2.x()-xmin, v2.y()-ymin, v3.x()-xmin, v3.y()-ymin, 1);

	for (int y=0; y<nh; y++){
		for (int x=0; x<nw; x++){
			if (pMask.get()[y*nw+x] == 0){
				pMask.get()[y*nw+x] = 1;
			}
			else{
				break;
			}
		}
		for (int x=nw-1; x>=0; x--){
			if (pMask.get()[y*nw+x] == 0){
				pMask.get()[y*nw+x] = 1;
			}
			else{
				break;
			}
		}
	}

	for (int y=0; y<nh; y++){
		int yIdx = y+ymin;
		if (yIdx >= nHeight || yIdx < 0){
			continue;
		}
		for (int x=0; x<nw; x++){
			int xIdx = x + xmin;
			if (xIdx >= nWidth || xIdx < 0){
				continue;
			}
			if (pMask.get()[y*nw+x] != 1){
				if (pZBuffer[yIdx*nWidth+xIdx] <= 0 || pZBuffer[yIdx*nWidth+xIdx] < zBuffer){
					pImage[yIdx*nWidth+xIdx] = diffuese;
					pZBuffer[yIdx*nWidth+xIdx] = zBuffer;
				}
			}
		}
	}

}

template <class T>
bool Methods::InBox(Point<T,2> pt, int left, int up, int right, int down)
{
	return (pt.x()>=left && pt.x()<=right && pt.y()>=up && pt.y()<=down);
}

template <class T>
std::vector< Point<T,2> > Methods::KeepClosed(std::vector< Point<T,2> > contour)
{
	if (contour.size() <= 0){
		return contour;
	}
	contour.push_back(contour[0]);
	std::vector< Point<T,2> > contourClosed;
	contourClosed.push_back(contour[0]);
	for (size_t i=1; i<contour.size(); i++){
		Point<T,2> ptBegin = contourClosed.back();
		Point<T,2> ptEnd = contour[i];
		double len = ptBegin.DistanceTo(ptEnd);
		if (len <= 1.0f){
			contourClosed.push_back(ptEnd);
		}
		else {
			Direction<T,2> dir(ptBegin, ptEnd);
			for (int l=1; l<len; l++){
				contourClosed.push_back(ptBegin + dir*l);
			}
			contourClosed.push_back(ptEnd);
		}
	}

	return contourClosed;
}

template <class T>
std::vector< Point<T,2> > Methods::InterpAndInBox(std::vector< Point<T,2> > contour, int left, int up, int right, int down)
{
	if (contour.size() <= 0){
		return contour;
	}
	contour.push_back(contour[0]);

	std::vector< Point<T,2> > contourClosed;
	if (contour[0].x()>=left && contour[0].x()<=right && contour[0].y()>=up && contour[0].y()<=down){
		contourClosed.push_back(contour[0]);	
	}
	Point<T,2> ptBegin = contour[0];
	Point<T,2> ptEnd;
	for (size_t i=1; i<contour.size(); i++){
		ptEnd = contour[i];
		double len = ptBegin.DistanceTo(ptEnd);
		Direction<T,2> dir(ptBegin, ptEnd);
		for (int l=1; l<len; l++){
			if (Methods::InBox(ptBegin + dir*l, left, up, right, down)){
				contourClosed.push_back(ptBegin + dir*l);
			}
		}
		if (Methods::InBox(ptEnd, left, up, right, down)){
			contourClosed.push_back(ptEnd);
		}
		ptBegin = ptEnd;
	}

	return contourClosed;
}

std::shared_ptr<unsigned char> Methods::Contour2dToMask(std::vector< Point2d > contour, int nWidth, int nHeight, bool withContour/*=true*/){
	return Methods::ContourToMask(contour, nWidth, nHeight, withContour);
}

std::shared_ptr<unsigned char> Methods::Contour2fToMask(std::vector< Point2f > contour, int nWidth, int nHeight, bool withContour/*=true*/){
	return Methods::ContourToMask(contour, nWidth, nHeight, withContour);
}

template <class T>
std::shared_ptr<unsigned char> Methods::ContourToMask(std::vector< Point<T,2> > contour, int nWidth, int nHeight, bool withContour)
{
	std::shared_ptr<unsigned char> pMask(new unsigned char[nWidth*nHeight]);
	memset(pMask.get(), 0, nWidth*nHeight*sizeof(unsigned char));

	//封闭曲线
	std::vector< Point<T,2> > contourInterpBox = Methods::InterpAndInBox(contour, 0, 0, nWidth-1, nHeight-1);
	std::vector< Point<T,2> > contourClosed = Methods::KeepClosed(contourInterpBox);

	//边界二值化 & 边缘扩充
	int nWidthExt = nWidth + 2;
	int nHeightExt = nHeight + 2;
	std::shared_ptr<unsigned char> pMaskExt(new unsigned char[nWidthExt*nHeightExt]);
	memset(pMaskExt.get(), 0, nWidthExt*nHeightExt*sizeof(unsigned char));
	int xMin=nWidthExt, xMax=-1, yMin=nHeightExt, yMax=-1;
	for(size_t i=0; i<contourClosed.size(); i++){
		int x = int(contourClosed[i].x() + 0.5);
		int y = int(contourClosed[i].y() + 0.5);
		x = x>=1 ? x:1;
		x = x<=nWidthExt-2 ? x:nWidthExt-2;
		y = y>=1 ? y:1;
		y = y<=nHeightExt-2 ? y:nHeightExt-2;

		xMin = xMin>x ? x:xMin;
		xMax = xMax<x ? x:xMax;
		yMin = yMin>y ? y:yMin;
		yMax = yMax<y ? y:yMax;

		pMaskExt.get()[(y+1)*nWidthExt+x+1] = 1;
	}

	std::vector< Point<int,2> > ptSeeds;
	//上下左右向中间扫描
	for (int y=0; y<nHeightExt; y++){
		for (int x=0; x<nWidthExt; x++){
			if (pMaskExt.get()[y*nWidthExt+x] > 0){
				break;
			}
			if (x >= xMin && y>=yMin && y<=yMax){
				ptSeeds.push_back(Point<int,2>(x, y));
			}
			pMaskExt.get()[y*nWidthExt+x] = 2;
		}
		for (int x=nWidthExt-1; x>=0; x--){
			if (pMaskExt.get()[y*nWidthExt+x] > 0){
				break;
			}
			if (x <= xMax && y>=yMin && y<=yMax){
				ptSeeds.push_back(Point<int,2>(x, y));
			}
			pMaskExt.get()[y*nWidthExt+x] = 2;
		}
	}
	for (int x=0; x<nWidthExt; x++){
		for (int y=0; y<nHeightExt; y++){
			if (pMaskExt.get()[y*nWidthExt+x] > 0){
				break;
			}
			if (y >= yMin && x>=xMin && x<=xMax){
				ptSeeds.push_back(Point<int,2>(x, y));
			}
			pMaskExt.get()[y*nWidthExt+x] = 2;
		}
		for (int y=nHeight-1; y>=0; y--){
			if (pMaskExt.get()[y*nWidthExt+x] > 0){
				break;
			}
			if (y <= yMax && x>=xMin && x<=xMax){
				ptSeeds.push_back(Point<int,2>(x, y));
			}
			pMaskExt.get()[y*nWidthExt+x] = 2;
		}
	}

	//根据种子点填充其它点
	std::vector< Point<int,2> > ptSeedsNext;
	while (ptSeeds.size() > 0)
	{
		ptSeedsNext.clear();
		for (size_t i=0; i<ptSeeds.size(); i++){
			int x = ptSeeds[i].x();
			int y = ptSeeds[i].y();
			if (pMaskExt.get()[y*nWidthExt+x-1] <= 0){
				ptSeedsNext.push_back(Point<int,2>(x-1, y));
				pMaskExt.get()[y*nWidthExt+x-1] = 2;
			}
			if (pMaskExt.get()[y*nWidthExt+x+1] <= 0){
				ptSeedsNext.push_back(Point<int,2>(x+1, y));
				pMaskExt.get()[y*nWidthExt+x+1] = 2;
			}
			if (pMaskExt.get()[(y-1)*nWidthExt+x] <= 0){
				ptSeedsNext.push_back(Point<int,2>(x, y-1));
				pMaskExt.get()[(y-1)*nWidthExt+x] = 2;
			}
			if (pMaskExt.get()[(y+1)*nWidthExt+x] <= 0){
				ptSeedsNext.push_back(Point<int,2>(x, y+1));
				pMaskExt.get()[(y+1)*nWidthExt+x] = 2;
			}
		}
		ptSeeds = ptSeedsNext;
	}

	//截取
	for (int y=0; y<nHeightExt; y++){
		for (int x=0; x<nWidthExt; x++){
			if (pMaskExt.get()[y*nWidthExt+x] == 0){
				pMask.get()[(y-1)*nWidth+x-1] = 2;
			}
			else if (pMaskExt.get()[y*nWidthExt+x] == 1 && withContour){
				pMask.get()[(y-1)*nWidth+x-1] = 1;
			}
		}
	}

	return pMask;
}

double Methods::Distance_Point2Line(Point3d pt, Direction3d dir, Point3d ptLine)
{
	double lengthVec = pt.DistanceTo(ptLine);
	double lengthProj = Methods::Length_VectorInLine(pt, dir, ptLine);
	return sqrt(lengthVec*lengthVec - lengthProj*lengthProj);
}

double Methods::Length_VectorInLine(Point3d pt, Direction3d dir, Point3d ptLine)
{
	Point3d vec = pt - ptLine;
	return vec.x()*dir.x() + vec.y()*dir.y() + vec.z()*dir.z();
}

Point3d Methods::Projection_Point2Line(Point3d pt, Direction3d dir, Point3d ptLine)
{
	double l = Methods::Length_VectorInLine(pt, dir, ptLine);
	return ptLine + dir * l;
}

double Methods::Distance_Point2Plane(Point3d pt, Direction3d dirH, Direction3d dirV, Point3d ptPlane)
{
	Direction3d dirNorm = dirH.cross(dirV);
	return Methods::Distance_Point2Plane(pt, dirNorm, ptPlane);
}

double Methods::Distance_Point2Plane(Point3d pt, Direction3d dirNorm, Point3d ptPlane)
{
	Point3d vec = pt - ptPlane;
	return vec.x()*dirNorm.x() + vec.y()*dirNorm.y() + vec.z()*dirNorm.z();
}

Point3d Methods::Projection_Point2Plane(Point3d pt, Direction3d dirH, Direction3d dirV, Point3d ptPlane)
{
	Direction3d dirNorm = dirH.cross(dirV);
	return Methods::Projection_Point2Plane(pt, dirNorm, ptPlane);
}

Point3d Methods::Projection_Point2Plane(Point3d pt, Direction3d dirNorm, Point3d ptPlane)
{
	double dist = Methods::Distance_Point2Plane(pt, dirNorm, ptPlane);
	return Point3d(pt.x()-dist*dirNorm.x(), pt.y()-dist*dirNorm.y(), pt.z()-dist*dirNorm.z());
}

Point3d Methods::RotatePoint3D(Point3d pt, Direction3d dirNorm, Point3d ptCenter, float theta)
{  
	Point3d ptOut;
	double dx = dirNorm.x();
	double dy = dirNorm.y();
	double dz = dirNorm.z();
	double x0 = pt.x() - ptCenter.x();
	double y0 = pt.y() - ptCenter.y();
	double z0 = pt.z() - ptCenter.z();

	ptOut.SetX(
		x0 * (dx * dx * (1 - cosf(theta)) + cosf(theta)) + 
		y0 * (dx * dy * (1 - cosf(theta)) - dz * sinf(theta)) + 
		z0 * (dx * dz * (1 - cosf(theta)) + dy * sinf(theta))
	);
	ptOut.SetY(
		x0 * (dy * dx * (1 - cosf(theta)) + dz * sinf(theta)) +  
		y0 * (dy * dy * (1 - cosf(theta)) + cosf(theta)) + 
		z0 * (dy * dz * (1 - cosf(theta)) - dx * sinf(theta))
	);
	ptOut.SetZ(
		x0 * (dz * dx * (1 - cosf(theta)) - dy * sinf(theta)) + 
		y0 * (dy * dz * (1 - cosf(theta)) + dx * sinf(theta)) + 
		z0 * (dz * dz * (1 - cosf(theta)) + cosf(theta))
	);

	return (ptOut + ptCenter);
}

int Methods::GetLengthofCrossLineInBox(Direction3d dir, double spacing, double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
{
	double xCenter = (xMin + xMax) / 2;
	double yCenter = (yMin + yMax) / 2;
	double zCenter = (zMin + zMax) / 2;

	int nUpPart = 0, nDownPart = 0;
	for (int i=0; ;i++)
	{
		double xTemp = xCenter + i*spacing*dir.x();
		double yTemp = yCenter + i*spacing*dir.y();
		double zTemp = zCenter + i*spacing*dir.z();
		if (xTemp<=xMin || xTemp>=xMax || yTemp<=yMin || yTemp>=yMax || zTemp<=zMin || zTemp>=zMax)
		{
			nUpPart = i;
			break;
		}
	}
	for (int i=0; ;i--)
	{
		double xTemp = xCenter + i*spacing*dir.x();
		double yTemp = yCenter + i*spacing*dir.y();
		double zTemp = zCenter + i*spacing*dir.z();
		if (xTemp<=xMin || xTemp>=xMax || yTemp<=yMin || yTemp>=yMax || zTemp<=zMin || zTemp>=zMax)
		{
			nDownPart = -i;
			break;
		}
	}
	return nUpPart + nDownPart;
}

Point3d Methods::GetTransferPoint(double m[3][3], Point3d pt)
{
	double r[3];
	for (int i=0; i<3; i++)
	{
		r[i] = 0;
		for (int j=0; j<3; j++)
		{
			r[i] += m[i][j]*pt[j];
		}
	}
	return Point3d(r[0], r[1], r[2]);
}

Point3f Methods::GetTransferPointf(float m[9], Point3f pt)
{
	double r[3];
	for (int i=0; i<3; i++)
	{
		r[i] = 0;
		for (int j=0; j<3; j++)
		{
			r[i] += m[i*3+j]*pt[j];
		}
	}
	return Point3f(r[0], r[1], r[2]);
}
