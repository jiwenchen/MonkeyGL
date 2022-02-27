#pragma once
#include <string>
#include <map>
#include "Defines.h"

namespace MonkeyGL{

    class TransferFunction
    {
    public:
        TransferFunction(void);
        ~TransferFunction(void);

    public:
        void SetMinPos(int pos){
            m_nMinPos = pos;
        }
        int GetMinPos(){
            return m_nMinPos;
        }

        void SetMaxPos(int pos){
            m_nMaxPos = pos;
        }
        int GetMaxPos(){
            return m_nMaxPos;
        }

        void SetControlPoints(std::map<int,RGBA> ctrlPts){
            m_pos2rgba = ctrlPts;
            m_pos2alpha.clear();
        }
        void SetControlPoints(std::map<int,RGBA> rgbPts, std::map<int, double> alphaPts){
            m_pos2rgba = rgbPts;
            m_pos2alpha = alphaPts;
        }

        void AddControlPoint(int pos, RGBA clr){
            m_pos2rgba[pos] = clr;
        }
        std::map<int, RGBA> GetControlPoints(){
            return m_pos2rgba;
        }

        bool GetTransferFunction(RGBA*& pBuffer, int& nLen);

    private:
        std::map<int, RGBA> m_pos2rgba;
        std::map<int, double> m_pos2alpha;
        int m_nMinPos;
        int m_nMaxPos;
    };

}


