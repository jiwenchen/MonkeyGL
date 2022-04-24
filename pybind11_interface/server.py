import imp
import sys
import numpy as np
import base64
import io

print (sys.path[0])
sys.path.append(f'{sys.path[0]}/build')
import pyMonkeyGL as mk

from typing import Optional
from fastapi import FastAPI
import uvicorn
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import SimpleITK as sitk

app = FastAPI()

hm = mk.HelloMonkey()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/volumetype')
def set_volume_type(
    data: dict
):
    vol_type = data.get('vol_type', 0)

    file_path = f'{sys.path[0]}/../data'
    vol_file = 'cardiac.raw'
    width = 512
    height = 512
    depth = 361
    spacing = (0.3510, 0.3510, 0.3)
    dirX = mk.Direction3d(1., 0., 0.)
    dirY = mk.Direction3d(0., 1., 0.)
    dirZ = mk.Direction3d(0., 0., -1.)

    tf = {}
    ww = 400
    wl = 40
    hm.SetColorBackground(mk.RGBA(0., 0.1, 0.1, 1.0))

    if vol_type == 1:
        tf[0] = mk.RGBA(0.8, 0, 0, 0)
        tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
        tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
        tf[99] = mk.RGBA(1, 0.8, 1, 1)
        ww = 500
        wl = 250
        hm.SetVolumeFile(f'{file_path}/{vol_file}', width, height, depth)
        hm.SetVRWWWL(ww, wl)
        hm.SetTransferFunc(tf)
        hm.SetDirection(dirX, dirY, dirZ)
    elif vol_type == 2:
        if 1:
            tf[0] = mk.RGBA(0.8, 0, 0, 0)
            tf[28] = mk.RGBA(0.8, 0.8, 0, 0.7)
            tf[99] = mk.RGBA(1, 0.8, 1, 1)
            ww = 420
            wl = 290
            depth = 1559
        else:
            tf[10] = mk.RGBA(0.3, 0.3, 0.8, 0)
            tf[20] = mk.RGBA(0.3, 0.6, 1, 0.3)
            tf[35] = mk.RGBA(0.3, 0.7, 0.8, 0)
            ww = 2000
            wl = 0
            depth = 559
        vol_file = 'body.raw'
        hm.SetVolumeFile(f'{file_path}/{vol_file}', width, height, depth)
        hm.SetVRWWWL(ww, wl)
        hm.SetTransferFunc(tf)
        spacing = (0.7422, 0.7422, 1.0)
        hm.SetDirection(dirX, dirY, dirZ)
    elif vol_type == 3:
        tf[0] = mk.RGBA(0.8, 0, 0, 0)
        tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
        tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
        tf[99] = mk.RGBA(1, 0.8, 1, 1)
        ww = 500
        wl = 250
        vol_file = 'rib.raw'
        depth = 521
        dirZ = mk.Direction3d(0., 0., 1.)
        hm.SetDirection(dirX, dirY, dirZ)
        hm.SetVolumeFile(f'{file_path}/{vol_file}', width, height, depth)
        hm.SetVRWWWL(ww, wl)
        hm.SetTransferFunc(tf)
        spacing = (0.8496089, 0.8496089, 0.625)
    elif vol_type == 4:
        itk_img = sitk.ReadImage(f'{file_path}/neckcta.nrrd')
        dir = itk_img.GetDirection()
        dirX = mk.Direction3d(dir[0], dir[1], dir[2])
        dirY = mk.Direction3d(dir[3], dir[4], dir[5])
        dirZ = mk.Direction3d(dir[6], dir[7], dir[8])
        spacing = itk_img.GetSpacing()
        depth = itk_img.GetDepth()
        npdata = sitk.GetArrayFromImage(itk_img)
        npdatat = npdata.swapaxes(2,0)

        itk_mask = sitk.ReadImage(f'{file_path}/neckcta_mask.nii.gz')
        npmaskdata = sitk.GetArrayFromImage(itk_mask)
        npmaskdatat = npmaskdata.swapaxes(2,0)

        hm.SetDirection(dirX, dirY, dirZ)
        hm.SetVolumeArray(npdatat)

        tf0 = {}
        tf0[10] = mk.RGBA(1.0, 1.0, 1.0, 0)
        tf0[90] = mk.RGBA(1.0, 1.0, 1.0, 1)
        ww0 = 300
        wl0 = 250
        hm.SetVRWWWL(ww0, wl0)
        hm.SetObjectAlpha(0.6, 0)
        hm.SetTransferFunc(tf0)

        hm.AddNewObjectMaskArray(npmaskdatat)
        tf1 = {}
        tf1[10] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww1 = 500
        wl1 = 150
        hm.SetVRWWWL(ww1, wl1)
        hm.SetObjectAlpha(0.7, 1)
        hm.SetTransferFunc(tf1)


    elif vol_type == 5:
        itk_img = sitk.ReadImage(f'{file_path}/corocta.nrrd')
        dir = itk_img.GetDirection()
        dirX = mk.Direction3d(dir[0], dir[1], dir[2])
        dirY = mk.Direction3d(dir[3], dir[4], dir[5])
        dirZ = mk.Direction3d(dir[6], dir[7], dir[8])
        spacing = itk_img.GetSpacing()
        depth = itk_img.GetDepth()
        npdata = sitk.GetArrayFromImage(itk_img)
        npdatat = npdata.swapaxes(2,0)

        itk_mask = sitk.ReadImage(f'{file_path}/corocta_vessel_mask.nii.gz')
        npmaskdata = sitk.GetArrayFromImage(itk_mask)
        npmaskdatat = npmaskdata.swapaxes(2,0)

        itk_mask2 = sitk.ReadImage(f'{file_path}/corocta_heart_mask.nii.gz')
        npmaskdata2 = sitk.GetArrayFromImage(itk_mask2)
        npmaskdatat2 = npmaskdata2.swapaxes(2,0)

        hm.SetDirection(dirX, dirY, dirZ)
        hm.SetVolumeArray(npdatat)

        tf0 = {}
        tf0[5] = mk.RGBA(0.8, 0.8, 0.8, 0)
        tf0[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww0 = 500
        wl0 = 100
        hm.SetVRWWWL(ww0, wl0)
        hm.SetObjectAlpha(0, 0)
        hm.SetTransferFunc(tf0)

        label1 = hm.AddNewObjectMaskArray(npmaskdatat)
        tf1 = {}
        tf1[5] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww1 = 500
        wl1 = 100
        hm.SetVRWWWL(ww1, wl1, 1)
        hm.SetObjectAlpha(1, 1)
        hm.SetTransferFunc(tf1)

        label2 = hm.AddNewObjectMaskArray(npmaskdatat2)
        tf1 = {}
        tf1[5] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww1 = 500
        wl1 = 100
        hm.SetVRWWWL(ww1, wl1, label2)
        hm.SetObjectAlpha(0.4, label2)
        hm.SetTransferFunc(tf1)
    
    hm.SetSpacing(spacing[0], spacing[1], spacing[2])

    return {
        'message': 'successful'
    }

@app.get('/vrdata')
def get_vr_data(
    x_angle: float,
    y_angle: float
):
    width = 512
    height = 512
    hm.Rotate(x_angle, y_angle)
    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/mprdata')
def get_mpr_data(
    plane_type: int
):
    b64str = hm.GetPlaneData_pngString(mk.PlaneType(plane_type))

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/mprbrowse')
def browse_mpr_data(
    plane_type: int,
    delta: float
):
    hm.Browse(delta, mk.PlaneType(plane_type))
    b64str = hm.GetPlaneData_pngString(mk.PlaneType(plane_type))

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/originbrowse')
def browse_origin_data(
    slice: int
):
    b64str = hm.GetOriginData_pngString(slice)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/updatethickness')
def update_thickness(
    thickness: float
):
    hm.UpdateThickness(thickness)

    return {
        'message': 'successful'
    }

@app.get('/updatemprtype')
def update_mpr_type(
    mpr_type: int
):
    hm.SetMPRType(mk.MPRType(mpr_type))

    return {
        'message': 'successful'
    }

if __name__ == "__main__":
    uvicorn.run(app='server:app', host="0.0.0.0", port=7788, reload=True, debug=True)
