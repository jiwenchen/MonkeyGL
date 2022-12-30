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
import uuid

base_path = Path(__file__).resolve().parent.parent
sys.path.append(str(base_path / "pybind11_interface" / "build"))

import pyMonkeyGL as mk

app = FastAPI()

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

monkeys = {}

def get_monkey_instance(
    uid: str
):
    return monkeys[uid]

@app.on_event('startup')
def init_data():
    volume["vol_type"] = 1
    return volume

@app.get('/initserver')
def init_server():
    uid = str(uuid.uuid4())
    monkeys[uid] = mk.HelloMonkey()
    # monkeys[uid].SetLogLevel(mk.LogLevelWarn)
    return {
        "uid": uid,
        'message': 'successful'
    }

@app.get('/releaseserver')
def release_server(
    uid: str
):
    msg = 'successful'
    if uid in monkeys:
        del monkeys[uid]
    else:
        msg = f'not exist uid[{uid}]'

    return {
        'message': msg
    }

@app.get('/releaseallserver')
def release_all_server(
):
    ks = tuple(monkeys.keys())
    for k in ks:
        del monkeys[k]

    return {
        'message': 'successful'
    }

@app.post('/volumetype')
def set_volume_type(
        data: dict
):
    vol_type = data.get('vol_type', 0)
    uid = data.get('uid', '')
    volume["vol_type"] = vol_type
    hm = get_monkey_instance(uid)

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
        # tf[10] = mk.RGBA(0.8, 0.8, 0.6, 0.0)
        # tf[99] = mk.RGBA(1, 0.8, 1, 1)

        # tf[10] = mk.RGBA(0.8, 0.8, 0.8, 0.0)
        # tf[90] = mk.RGBA(0.8, 0.8, 0.8, 0.8)

        tf[10] = mk.RGBA(0.3, 0.3, 0.4, 0)
        tf[20] = mk.RGBA(0.3, 0.8, 1, 0.5)
        tf[35] = mk.RGBA(0.3, 0.7, 0.8, 0)
        ww = 1000
        wl = -400

        # ww = 500
        # wl = 350
        vol_file = 'lung.nii.gz'
        hm.LoadVolumeFile(f'{file_path}/{vol_file}')
        hm.SetVRWWWL(ww, wl)
        hm.SetTransferFunc(tf)
        hm.SetObjectAlpha(0.8)

        label1 = hm.AddObjectMaskFile(f'{file_path}/lung_vein.nii.gz')
        tf1 = {}
        tf1[5] = mk.RGBA(0.8, 0, 0, 0)
        tf1[90] = mk.RGBA(0.8, 0.4, 0.8, 0.8)
        ww1 = 500
        wl1 = -700
        hm.SetVRWWWL(ww1, wl1, label1)
        hm.SetObjectAlpha(0.7, label1)
        hm.SetTransferFunc(tf1)

        label2 = hm.AddObjectMaskFile(f'{file_path}/lung_artery.nii.gz')
        tf2 = {}
        tf2[5] = mk.RGBA(0.8, 0.4, 0, 0.2)
        tf2[90] = mk.RGBA(0.0, 0.8, 0, 0.8)
        ww2 = 500
        wl2 = -700
        hm.SetVRWWWL(ww2, wl2, label2)
        hm.SetObjectAlpha(0.7, label2)
        hm.SetTransferFunc(tf2)

        label3 = hm.AddObjectMaskFile(f'{file_path}/lung_bronchia.nii.gz')
        tf3 = {}
        tf3[5] = mk.RGBA(1.0, 0.8, 1.0, 0.5)
        tf3[90] = mk.RGBA(0.0, 0.8, 0.9, 1.0)
        ww3 = 700
        wl3 = -1000
        hm.SetVRWWWL(ww3, wl3, label3)
        hm.SetObjectAlpha(0.8, label3)
        hm.SetTransferFunc(tf3)
        

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

    hm.SetVRSize(512, 512)

    return {
        'message': 'successful'
    }


@app.post('/lineindex')
def set_line_index(
        data: dict
):
    uid = data.get('uid', '')
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
            cpr_line = np.array([])
    hm = get_monkey_instance(uid)
    hm.SetCPRLinePatientArray(cpr_line)

    return {
        'message': 'successful'
    }

@app.get('/rotatevr')
def rotate_vr(
        uid: str,
        x_angle: float,
        y_angle: float
):
    hm = get_monkey_instance(uid)
    hm.Rotate(x_angle, y_angle)
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/panvr')
def pan_vr(
        uid: str,
        x_shift: float,
        y_shift: float
):
    hm = get_monkey_instance(uid)
    hm.Pan(x_shift, y_shift)
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/zoomvr')
def zoom_vr(
        uid: str,
        delta: float
):
    hm = get_monkey_instance(uid)
    hm.Zoom(delta)
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/reset')
def reset(
    uid: str
):
    hm = get_monkey_instance(uid)
    hm.Reset()
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/vrdata')
def get_vr_data(
    uid: str
):
    hm = get_monkey_instance(uid)
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/wwwl')
def set_ww_wl(
        uid: str,
        ww: float,
        wl: float
):
    hm = get_monkey_instance(uid)
    hm.SetVRWWWL(ww, wl)
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/loadtf')
def loadtf(
    uid: str
):
    hm = get_monkey_instance(uid)
    hm.LoadTransferFunction(f'{base_path}/trfns/coro.txt')

    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/savetf')
def savetf(
    uid: str
):
    hm = get_monkey_instance(uid)
    hm.SaveTransferFunction(f'{base_path}/trfns/test.txt')
    
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/setrendertype')
def set_render_type(
    uid: str,
    type: int
):
    hm = get_monkey_instance(uid)
    hm.SetRenderType(mk.RenderType(type))
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }

@app.get('/orientation')
def orientation(
    uid: str,
    dir: int
):
    hm = get_monkey_instance(uid)
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

    hm.ShowPlaneInVR(True)
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/vrenablecprline')
def vrenablecprline(
        uid: str,
        enableCPR: bool
):
    hm = get_monkey_instance(uid)
    hm.EnableLayer(mk.LayerTypeCPRLine, enableCPR)
    hm.SetCPRLineColor(mk.RGBA(1.0, 1.0, 0.0))
    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/mprdata')
def get_mpr_data(
        uid: str,
        plane_type: int
):
    hm = get_monkey_instance(uid)
    b64str = hm.GetPlaneData_pngString(mk.PlaneType(plane_type))
    pt = hm.GetCrossHairPoint(mk.PlaneType(plane_type))

    return {
        'data': {
            "x": pt.x(),
            "y": pt.y(),
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/rotatestretchedcpr')
def get_stretched_cpr(
        uid: str,
        angle: float,
):
    hm = get_monkey_instance(uid)
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
        uid: str,
        angle: float,
):
    hm = get_monkey_instance(uid)
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
        uid: str,
        plane_type: int,
        delta: float
):
    hm = get_monkey_instance(uid)
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
        uid: str,
        slice: int
):
    hm = get_monkey_instance(uid)
    b64str = hm.GetOriginData_pngString(slice)

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


@app.get('/updatethickness')
def update_thickness(
        uid: str,
        thickness: float
):
    hm = get_monkey_instance(uid)
    hm.UpdateThickness(thickness)

    return {
        'message': 'successful'
    }

@app.get('/setthickness')
def set_thickness(
        uid: str,
        plane_type: int,
        thickness: float
):
    hm = get_monkey_instance(uid)
    hm.SetThickness(thickness, mk.PlaneType(plane_type))

    return {
        'message': 'successful'
    }


@app.get('/updatemprtype')
def update_mpr_type(
        uid: str,
        mpr_type: int
):
    hm = get_monkey_instance(uid)
    hm.SetMPRType(mk.MPRType(mpr_type))

    return {
        'message': 'successful'
    }

@app.get('/panmpr')
def pan_mpr(
        uid: str,
        plane_type: int,
        x: float,
        y: float
):
    hm = get_monkey_instance(uid)
    hm.PanCrossHair(x, y, mk.PlaneType(plane_type))

    return {
        'message': 'successful'
    }

@app.get('/rotatech')
def rotate_cross_hair(
        uid: str,
        plane_type: int,
        angle: float
):
    hm = get_monkey_instance(uid)
    hm.RotateCrossHair(angle, mk.PlaneType(plane_type))

    return {
        'message': 'successful'
    }

@app.get('/showplaneinvr')
def show_plane_in_vr(
    uid: str,
    show: bool
):
    hm = get_monkey_instance(uid)
    hm.ShowPlaneInVR(show)

    return {
        'message': 'successful'
    }

@app.get('/showannotation')
def show_annotation(
    uid: str,
    show: bool
):
    hm = get_monkey_instance(uid)
    if show:
        # hm.AddAnnotation(mk.PlaneVR, "Hello - Monkey@?iJ", 250, 100, mk.FontSizeSmall, mk.AnnotationFormatLeft, mk.RGBA(1.0, 0.0, 0.0))
        # hm.AddAnnotation(mk.PlaneVR, "Hello - Monkey@?iJ", 250, 300, mk.FontSizeMiddle, mk.AnnotationFormatCenter, mk.RGBA(0.0, 1.0, 0.0))
        hm.AddAnnotation(mk.PlaneVR, "abcdefghiljkl mnopqrstuvwxyz", 500, 120, mk.FontSizeBig, mk.AnnotationFormatRight, mk.RGBA(1.0, 0.0, 0.0))
        hm.AddAnnotation(mk.PlaneVR, "ABCDEFGHIJK LMN", 500, 30, mk.FontSizeMiddle, mk.AnnotationFormatRight, mk.RGBA(1.0, 0.0, 1.0))
        hm.AddAnnotation(mk.PlaneVR, "LMNOPQRSTUVWX YZ", 500, 80, mk.FontSizeBig, mk.AnnotationFormatRight, mk.RGBA(1.0, 1.0, 0.0))
        hm.AddAnnotation(mk.PlaneVR, '''~`·!@#$%^&*()_-+=[]{}\\|;''', 500, 180, mk.FontSizeSmall, mk.AnnotationFormatRight, mk.RGBA(0.0, 1.0, 1.0))
        hm.AddAnnotation(mk.PlaneVR, ''';\':",./<>? 1234567890''', 500, 220, mk.FontSizeBig, mk.AnnotationFormatRight, mk.RGBA(0.6, 0.0, 1.0))

        # hm.RemovePlaneAnnotations(mk.PlaneVR)
        # hm.RemoveAllAnnotations()
    hm.EnableLayer(mk.LayerTypeAnnotation, show)

    b64str = hm.GetVRData_pngString()

    return {
        'data': {
            'image': b64str
        },
        'message': 'successful'
    }


if __name__ == "__main__":
    uvicorn.run(app='server:app', host="0.0.0.0",
                port=7788, reload=True, debug=True)
