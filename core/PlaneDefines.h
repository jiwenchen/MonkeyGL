#pragma once

namespace MonkeyGL{
    
    enum ePlaneType
    {
        ePlaneType_NULL = -1,
        ePlaneType_Axial,
        ePlaneType_Sagittal,
        ePlaneType_Coronal,
        ePlaneType_Axial_Oblique,
        ePlaneType_Sagittal_Oblique,
        ePlaneType_Coronal_Oblique,
        ePlaneType_Batch,
        ePlaneType_VolumeRender,

        ePlaneType_Count
    };

    struct OrthoMPRParameters
    {
        ePlaneType planeType;

        OrthoMPRParameters(){
            planeType = ePlaneType_NULL;
        }
    };

}
