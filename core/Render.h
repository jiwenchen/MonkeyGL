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
#include "Defines.h"
#include "TransferFunction.h"
#include "VolumeInfo.h"
#include "Methods.h"
#include "IRender.h"
#include "CuDataInfo.h"

namespace MonkeyGL {

    class Render : public IRender
    {
    public:
        Render(void);
        ~Render(void);

    public:
    // volume info
        virtual bool SetVolumeData(std::shared_ptr<short>pData, int nWidth, int nHeight, int nDepth);
        virtual unsigned char AddNewObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth);
        virtual unsigned char AddObjectMaskFile(const char* szFile);
        virtual bool UpdateObjectMask(std::shared_ptr<unsigned char>pData, int nWidth, int nHeight, int nDepth, const unsigned char& nLabel);
        virtual void LoadVolumeFile(const char* szFile);
        virtual void SetSpacing(double x, double y, double z);
        virtual void Reset();
        
        virtual bool TransferVoxel2ImageInVR(float& fx, float& fy, int nWidth, int nHeight, Point3d ptVoxel);

    // output
        virtual bool GetPlaneMaxSize(int& nWidth, int& nHeight, const PlaneType& planeType);
        virtual bool GetPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);

        virtual bool GetCrossHairPoint(double& x, double& y, const PlaneType& planeType);
        virtual void PanCrossHair(float fx, float fy, PlaneType planeType);

        virtual bool GetVRData(unsigned char* pVR, int nWidth, int nHeight);

        virtual bool GetBatchData( std::vector<short*>& vecBatchData, BatchInfo batchInfo );

        virtual bool GetPlaneRotateMatrix(float* pMatirx, PlaneType planeType);


        virtual void Anterior();
        virtual void Posterior();
        virtual void Left();
        virtual void Right();
        virtual void Head();
        virtual void Foot();

        virtual void SetRenderType(RenderType type){
            m_renderType = type;
        }
        virtual void Rotate(float fxRotate, float fyRotate);
        virtual float Zoom(float ratio);
        virtual float GetZoomRatio();
        virtual void Pan(float fxShift, float fyShift);
        
        virtual bool SetVRWWWL(float fWW, float fWL);
        virtual bool SetVRWWWL(float fWW, float fWL, unsigned char nLabel);
        virtual bool SetObjectAlpha(float fAlpha);
        virtual bool SetObjectAlpha(float fAlpha, unsigned char nLabel);
        virtual bool SetTransferFunc(std::map<int, RGBA> ctrlPts);
        virtual bool SetTransferFunc(std::map<int, RGBA> ctrlPts, unsigned char nLabel);
        virtual bool SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts);
        virtual bool SetTransferFunc(std::map<int, RGBA> rgbPts, std::map<int, float> alphaPts, unsigned char nLabel);
        virtual bool LoadTransferFunction(const char* szFile);

        void ShowPlaneInVR(bool bShow);

    private:
        void Init();
        void UpdateTransferFunctions();
        void UpdateAlphaWWWL();
        void NormalizeVOI();

        bool GetMPRPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);
        bool GetCPRPlaneData(std::shared_ptr<short>& pData, int& nWidth, int& nHeight, const PlaneType& planeType);

        void testcuda();

        void InitCommon(float fxSpacing, float fySpacing, float fzSpacing, cudaExtent volumeSize){
            m_f3Spacing.x = fxSpacing;
            m_f3Spacing.y = fySpacing;
            m_f3Spacing.z = fzSpacing;
            m_f3SpacingVoxel.x = 1.0f / volumeSize.width;
            m_f3SpacingVoxel.y = 1.0f / volumeSize.height;
            m_f3SpacingVoxel.z = 1.0f / volumeSize.depth;

            float fMaxSpacing = max(fxSpacing, max(fySpacing, fzSpacing));	

            float fMaxLen = max(volumeSize.width*fxSpacing, max(volumeSize.height*fySpacing, volumeSize.depth*fzSpacing));
            m_f3maxLenSpacing.x = 1.0f*fMaxLen/(volumeSize.width*fxSpacing);
            m_f3maxLenSpacing.y = 1.0f*fMaxLen/(volumeSize.height*fySpacing);
            m_f3maxLenSpacing.z = 1.0f*fMaxLen/(volumeSize.depth*fzSpacing);
        }

    private:
        float m_fVOI_xStart;
        float m_fVOI_xEnd;
        float m_fVOI_yStart;
        float m_fVOI_yEnd;
        float m_fVOI_zStart;
        float m_fVOI_zEnd;
        VOI m_voi_Normalize;

        RenderType m_renderType;

        float m_fTotalXTranslate;
        float m_fTotalYTranslate;
        float m_fTotalScale;

        AlphaAndWWWLInfo m_AlphaAndWWWLInfo;
        float m_pRotateMatrix[9];
        float m_pTransposRotateMatrix[9];
        float m_pTransformMatrix[9];
        float m_pTransposeTransformMatrix[9];

        CuDataInfo m_cuDataInfo;
        float3 m_f3SpacingVoxel;
        float3 m_f3Spacing;
        float3 m_f3maxLenSpacing;
        cudaExtent m_VolumeSize;
    };
}


