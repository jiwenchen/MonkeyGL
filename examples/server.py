import SimpleITK as sitk
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from enum import Enum
import uvicorn
from fastapi import FastAPI
from typing import Optional
import sys
import numpy as np
import json
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent
sys.path.append(str(base_path / "pybind11_interface" / "build"))

import pyMonkeyGL as mk

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


html_path = str(base_path / "examples" / "html")
app.mount("/hm", StaticFiles(directory=html_path), name="hm")


volume = {"vol_type": 0}


@app.on_event('startup')
def init_data():
    volume["vol_type"] = 1
    return volume


@app.post('/volumetype')
def set_volume_type(
        data: dict
):
    vol_type = data.get('vol_type', 0)
    volume["vol_type"] = vol_type

    file_path = f'{sys.path[0]}/../data'
    vol_file = 'cardiac.mhd'

    tf = {}
    hm.SetColorBackground(mk.RGBA(0., 0.1, 0.1, 1.0))

    if vol_type == 1:
        tf[0] = mk.RGBA(0.8, 0, 0, 0)
        tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
        tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
        tf[99] = mk.RGBA(1, 0.8, 1, 1)
        ww = 500
        wl = 250
        hm.LoadVolumeFile(f'{file_path}/{vol_file}')
        hm.SetVRWWWL(ww, wl)
        hm.SetTransferFunc(tf)
    elif vol_type == 2:
        if 0:
            tf[0] = mk.RGBA(0.8, 0, 0, 0)
            tf[28] = mk.RGBA(0.8, 0.8, 0, 0.7)
            tf[99] = mk.RGBA(1, 0.8, 1, 1)
            ww = 420
            wl = 290
        else:
            tf[10] = mk.RGBA(0.3, 0.3, 0.4, 0)
            tf[20] = mk.RGBA(0.3, 0.8, 1, 0.5)
            tf[35] = mk.RGBA(0.3, 0.7, 0.8, 0)
            ww = 1200
            wl = -300

        vol_file = 'body.mhd'
        hm.LoadVolumeFile(f'{file_path}/{vol_file}')
        hm.SetVRWWWL(ww, wl)
        hm.SetTransferFunc(tf)

        npmaskdatat = np.zeros((1559, 512, 512))
        npmaskdatat[200:225, 200:240, 150:180] = 1

        hm.AddNewObjectMaskArray(npmaskdatat)
        tf1 = {}
        tf1[5] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww1 = 500
        wl1 = -500
        hm.SetVRWWWL(ww1, wl1)
        hm.SetObjectAlpha(1, 1)
        hm.SetTransferFunc(tf1)

    elif vol_type == 3:
        tf[0] = mk.RGBA(0.8, 0, 0, 0)
        tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
        tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
        tf[99] = mk.RGBA(1, 0.8, 1, 1)
        ww = 500
        wl = 250
        vol_file = 'rib.mhd'
        hm.LoadVolumeFile(f'{file_path}/{vol_file}')
        hm.SetVRWWWL(ww, wl)
        hm.SetTransferFunc(tf)
    elif vol_type == 4:
        itk_mask = sitk.ReadImage(f'{file_path}/neckcta_mask.nii.gz')
        npmaskdata = sitk.GetArrayFromImage(itk_mask)
        npmaskdatat = npmaskdata.swapaxes(2, 0)

        hm.LoadVolumeFile(f'{file_path}/neckcta.nrrd')

        tf0 = {}
        tf0[10] = mk.RGBA(1.0, 1.0, 1.0, 0)
        tf0[50] = mk.RGBA(1.0, 1.0, 1.0, 0.5)
        tf0[90] = mk.RGBA(1.0, 1.0, 1.0, 0.8)
        ww0 = 300
        wl0 = 250
        hm.SetVRWWWL(ww0, wl0)
        hm.SetObjectAlpha(0.4, 0)
        hm.SetTransferFunc(tf0)

        hm.AddNewObjectMaskArray(npmaskdatat)
        tf1 = {}
        tf1[5] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww1 = 500
        wl1 = 100
        hm.SetVRWWWL(ww1, wl1)
        hm.SetObjectAlpha(1, 1)
        hm.SetTransferFunc(tf1)

    elif vol_type == 5:
        # itk_mask = sitk.ReadImage(f'{file_path}/corocta_vessel_mask.nii.gz')
        # npmaskdata = sitk.GetArrayFromImage(itk_mask)
        # npmaskdatat = npmaskdata.swapaxes(2, 0)

        # itk_mask2 = sitk.ReadImage(f'{file_path}/corocta_heart_mask.nii.gz')
        # npmaskdata2 = sitk.GetArrayFromImage(itk_mask2)
        # npmaskdatat2 = npmaskdata2.swapaxes(2, 0)

        hm.LoadVolumeFile(f'{file_path}/corocta.nrrd')
        tf0 = {}
        tf0[5] = mk.RGBA(0.8, 0.8, 0.8, 0)
        tf0[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww0 = 500
        wl0 = 100
        hm.SetVRWWWL(ww0, wl0)
        hm.SetObjectAlpha(0, 0)
        hm.SetTransferFunc(tf0)

        # label2 = hm.AddNewObjectMaskArray(npmaskdatat2)
        label2 = hm.AddObjectMaskFile(f'{file_path}/corocta_heart_mask.nii.gz')
        tf1 = {}
        tf1[5] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww1 = 500
        wl1 = 100
        hm.SetVRWWWL(ww1, wl1, label2)
        hm.SetObjectAlpha(0.4, label2)
        hm.SetTransferFunc(tf1)

        # label1 = hm.AddNewObjectMaskArray(npmaskdatat)
        label1 = hm.AddObjectMaskFile(f'{file_path}/corocta_vessel_mask.nii.gz')
        tf1 = {}
        tf1[5] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)
        ww1 = 500
        wl1 = 100
        hm.SetVRWWWL(ww1, wl1, label1)
        hm.SetObjectAlpha(1, label1)
        hm.SetTransferFunc(tf1)

    return {
        'message': 'successful'
    }


@app.post('/lineindex')
def set_line_index(
        data: dict
):
    line_index = data.get('line_index', 0)
    vol_type = volume.get("vol_type", 0)
    cpr_line = np.array([])
    if vol_type == 5:
        with open(f'{base_path}/data/corocta.json', 'r') as f:
            lines = json.load(f)
        
        if line_index == 1:
            cpr_line = np.array(lines.get('line1', []))
        elif line_index == 2:
            cpr_line = np.array(lines.get('line2', []))
        
    elif vol_type == 4:
        with open(f'{base_path}/data/neckcta.json', 'r') as f:
            lines = json.load(f)
        if line_index == 1:
            cpr_line = np.array(lines.get('line1', []))
        elif line_index == 2:
            cpr_line = np.array(
            )
        
    hm.SetCPRLinePatientArray(cpr_line)

    return {
        'message': 'successful'
    }


@app.get('/rotatevr')
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


@app.get('/panvr')
def get_vr_data(
        x_shift: float,
        y_shift: float
):
    width = 512
    height = 512
    hm.Pan(x_shift, y_shift)
    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/zoomvr')
def zoom_vr(
        delta: float
):
    width = 512
    height = 512
    print(hm.Zoom(delta))
    print(hm.GetZoomRatio())
    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/reset')
def reset():
    width = 512
    height = 512
    hm.Reset()
    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/loadtf')
def loadtf():
    width = 512
    height = 512
    hm.LoadTransferFunction(f'{base_path}/trfns/coro.txt')

    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/savetf')
def savetf():
    width = 512
    height = 512
    hm.SaveTransferFunction(f'{base_path}/trfns/test.txt')
    
    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/switchvrmip')
def switchvrmip(
    vrmip: bool
):
    width = 512
    height = 512
    if vrmip:
        hm.EnableVR()
    else:
        hm.EnableMIP()
    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/orientation')
def orientation(
    dir: int
):
    width = 512
    height = 512
    if mk.OrientationAnterior == dir:
        hm.Anterior()
    elif mk.OrientationPosterior == dir:
        hm.Posterior()
    elif mk.OrientationLeft == dir:
        hm.Left()
    elif mk.OrientationRight == dir:
        hm.Right()
    elif mk.OrientationHead == dir:
        hm.Head()
    elif mk.OrientationFoot == dir:
        hm.Foot()
    b64str = hm.GetVRData_pngString(width, height)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/vrenablecprline')
def vrenablecprline(
        enableCPR: bool
):
    width = 512
    height = 512
    hm.ShowCPRLineInVR(enableCPR)
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


@app.get('/rotatestretchedcpr')
def get_stretched_cpr(
        angle: float,
):
    hm.RotateCPR(angle, mk.PlaneStretchedCPR)
    b64str = hm.GetPlaneData_pngString(mk.PlaneStretchedCPR)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/rotatestraightenedcpr')
def get_straightened_cpr(
        angle: float,
):
    hm.RotateCPR(angle, mk.PlaneStraightenedCPR)
    b64str = hm.GetPlaneData_pngString(mk.PlaneStraightenedCPR)

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

    print( hm.GetPlaneCurrentIndex(mk.PlaneType(plane_type)), hm.GetPlaneTotalNumber(mk.PlaneType(plane_type)))

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

@app.get('/panmpr')
def pan_mpr(
        plane_type: int,
        x: int,
        y:int
):
    hm.PanCrossHair(x, y, mk.PlaneType(plane_type))

    return {
        'message': 'successful'
    }


if __name__ == "__main__":
    uvicorn.run(app='server:app', host="0.0.0.0",
                port=7788, reload=True, debug=True)
