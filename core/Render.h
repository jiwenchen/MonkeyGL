#pragma once
#include "Defines.h"
#include "TransferFunction.h"
#include "VolumeInfo.h"
#include "Methods.h"
#include "IRender.h"

namespace MonkeyGL {

    class Render : public IRender
    {
    public:
        Render(void);
        ~Render(void);

    public:
    // volume info
        virtual void SetVolumeFile(const char* szFile, int nWidth, int nHeight, int nDepth);
        virtual void SetAnisotropy(double x, double y, double z);

        virtual void SetTransferFunc(const std::map<int, RGBA>& ctrlPoints);
        virtual void SetTransferFunc(const std::map<int, RGBA>& rgbPoints, const std::map<int, double>& alphaPoints);

    // output
        virtual bool GetPlaneMaxSize(int& nWidth, int& nHeight, const ePlaneType& planeType);
        virtual bool GetPlaneData(short* pData, int& nWidth, int& nHeight, const ePlaneType& planeType);

        virtual bool GetCrossHairPoint(double& x, double& y, const ePlaneType& planeType);
        virtual void PanCrossHair(int nx, int ny, ePlaneType planeType);

        virtual bool GetVRData(unsigned char* pPixelData, int nWidth, int nHeight);

        virtual bool GetBatchData( std::vector<short*>& vecBatchData, BatchInfo batchInfo );

        virtual bool GetPlaneRotateMatrix(float* pMatirx, ePlaneType planeType);


        virtual void Anterior();
        virtual void Posterior();
        virtual void Left();
        virtual void Right();
        virtual void Head();
        virtual void Foot();

        virtual void Rotate(float fxRotate, float fyRotate);
        virtual void Zoom(float ratio);
        virtual void Pan(float fxShift, float fyShift);
        virtual void SetWL(float fWW, float fWL);

    private:
        void CopyTransferFunc2Device();
        void NormalizeVOI();

    private:
        float m_fVOI_xStart;
        float m_fVOI_xEnd;
        float m_fVOI_yStart;
        float m_fVOI_yEnd;
        float m_fVOI_zStart;
        float m_fVOI_zEnd;
        VOI m_voi_Normalize;

        float m_fTotalXTranslate;
        float m_fTotalYTranslate;
        float m_fTotalScale;
        float m_fWW;
        float m_fWL;

        float* m_pRotateMatrix;
        float* m_pTransposRotateMatrix;
        float* m_pTransformMatrix;
        float* m_pTransposeTransformMatrix;
    };
}


