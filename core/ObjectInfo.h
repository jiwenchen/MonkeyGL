#pragma once
#include <map>
#include <fstream>
#include "Defines.h"
#include "TransferFunction.h"

namespace MonkeyGL{

    class ObjectInfo
    {
    public:
        float alpha;
        float ww;
        float wl;

        std::map<int, RGBA> idx2rgba;
        std::map<int, float> idx2alpha;

        ObjectInfo(){
            alpha = 1.0f;
            ww = 400.0f;
            wl = 40.0f;
            idx2rgba[5] = RGBA(0.8, 0.8, 0.8, 0);
            idx2rgba[95] = RGBA(0.8, 0.8, 0.8, 0.8);
            idx2alpha.clear();
        }

        ObjectInfo(float a, float w, float l){
            alpha = a;
            ww = w;
            wl = l;
            idx2rgba[5] = RGBA(0.8, 0.8, 0.8, 0);
            idx2rgba[95] = RGBA(0.8, 0.8, 0.8, 0.8);
            idx2alpha.clear();
        }

        void Print();

        bool GetTransferFunction( std::shared_ptr<RGBA>& pBuffer, int& nLen ) {
            TransferFunction tf;
            tf.SetControlPoints(idx2rgba, idx2alpha);
            return tf.GetTransferFunction(pBuffer, nLen);
        }

        static bool WriteFile(const char* szFile, const ObjectInfo& info){
            std::ofstream ofs(szFile, ios_base::out);
            if (!ofs.is_open()){
                return false;
            }
            std::map<int, RGBA> idx2rgba = info.idx2rgba;
            std::map<int, float> idx2alpha = info.idx2alpha;
            ofs << "ww: " << info.ww << std::endl;
            ofs << "wl: " << info.wl << std::endl;

            ofs << "idx2rgba: " << idx2rgba.size() << std::endl;
            for (std::map<int, RGBA>::iterator iter=idx2rgba.begin(); iter!=idx2rgba.end(); iter++){
                ofs << iter->first << " " << iter->second.red << " " << iter->second.green << " " << iter->second.blue << " " << iter->second.alpha << " " << std::endl;
            }

            ofs << "idx2alpha: " << idx2alpha.size() << std::endl;
            for (std::map<int, float>::iterator iter=idx2alpha.begin(); iter!=idx2alpha.end(); iter++){
                ofs << iter->first << " " << iter->second << " " << std::endl;
            }
            ofs.close();
            return true;
        }

        static bool ReadFile(const char* szFile, ObjectInfo& info){
            std::ifstream ifs(szFile, ios_base::in);
            if (!ifs.is_open()){
                return false;
            }
            std::string title;
            ifs >> title >> info.ww;
            ifs >> title >> info.wl;

            int numOfidx2rgba = 0;
            ifs >> title >> numOfidx2rgba;
            int idx;
            float red, green, blue, alpha;
            for (int i=0; i<numOfidx2rgba; i++){
                ifs >> idx >> red >> green >> blue >> alpha;
                info.idx2rgba[idx] = RGBA(red, green, blue, alpha);
            }

            int numofidx2alpha = 0;
            ifs >> title >> numofidx2alpha;
            for (int i=0; i<numofidx2alpha; i++){
                ifs >> idx >> alpha;
                info.idx2alpha[idx] = alpha;
            }
            
            ifs.close();
            return true;
        } 
    };

}