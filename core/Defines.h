#pragma once
#include <iostream>

using namespace std;

namespace MonkeyGL{

    #define PI 3.141592654

    struct RGBA
    {
        float red;
        float green;
        float blue;
        float alpha;

        RGBA(){
            red = 0.0f;
            green = 0.0f;
            blue = 0.0f;
            alpha = 0.0f;
        }
        RGBA(float r, float g, float b, float a){
            red = r;
            green = g;
            blue = b;
            alpha = a;
        }
        void Print(){
            cout << "red: " << red << endl; 
            cout << "green: " << green << endl; 
            cout << "blue: " << blue << endl; 
            cout << "alpha: " << alpha << endl; 
        }
    };

    struct VOI{
        int left;
        int right;
        int posterior;
        int anterior;
        int head;
        int foot;

        VOI() {
            left = -1;
            right = -1;
            posterior = -1;
            anterior = -1;
            head = -1;
            foot = -1;
        }
    };

    struct Orientation{
        bool bIsFF;
        float rx;
        float ry;
        float rz;
        float cx;
        float cy;
        float cz;
    } ;

    struct Lightparams{
        float ka;
        float ks;
        float kd;
        float lightColor[4];
        float globalAmbient[4];
    };

}