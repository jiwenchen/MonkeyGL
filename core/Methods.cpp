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

#include "Methods.h"
#include "math.h"
#include <iostream>
#include <cstring>
#include <memory>

using namespace MonkeyGL;

Methods::Methods(void)
{
}

Methods::~Methods(void)
{
}

void Methods::SetSeg( float* m, int n )
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
	Methods::SetSeg(Rx, 3);
	Methods::SetSeg(Rz, 3);
	Methods::SetSeg(Temp, 3);
	Methods::SetSeg(RxT, 3);
	Methods::SetSeg(RzT, 3);
	Methods::SetSeg(ST, 3);

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

void Methods::DrawDotInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, int x, int y, RGB clr)
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

void Methods::DrawLineInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, float x0, float y0, float x1, float y1, int nLineWidth, RGB clr){
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

void Methods::DrawCircleInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, float x, float y, float r, int nLineWidth, RGB clr){
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


void Methods::DrawTriangleInImage24Bit(unsigned char* pVR, int nWidth, int nHeight, Point2f v1, Point2f v2, Point2f v3, int nLineWidth, RGB clr)
{
	Methods::DrawLineInImage24Bit(pVR, nWidth, nHeight, v1.x(), v1.y(), v2.x(), v2.y(), nLineWidth, clr);
	Methods::DrawLineInImage24Bit(pVR, nWidth, nHeight, v2.x(), v2.y(), v3.x(), v3.y(), nLineWidth, clr);
	Methods::DrawLineInImage24Bit(pVR, nWidth, nHeight, v3.x(), v3.y(), v1.x(), v1.y(), nLineWidth, clr);
}

void Methods::FillHoleInImage24Bit(unsigned char* pVR, float* pZBuffer, int nWidth, int nHeight, Point2f v1, Point2f v2, Point2f v3, float zBuffer, RGB clr)
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
