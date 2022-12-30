import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent

sys.path.append(str(base_path / "pybind11_interface" / "build"))

import pyMonkeyGL as mk
import numpy as np
import time
import SimpleITK as sitk


file_path = str(base_path / "data")

def test_objs():
    e = mk.PlaneNotDefined

    p1 = mk.DeviceInfo()
    a = 10
    print (p1.GetCount(a))
    print (a)

    p2 = mk.RGBA(2, 3, 4, 1)
    p2.Print()

    d3 = mk.Direction3d(1.0, 1.0, 1.0)

def test_load_volme():
    hm = mk.HelloMonkey()
    # hm.SetLogLevel(mk.LogLevelWarn)
    dirX = mk.Direction3d(1., 0., 0.)
    dirY = mk.Direction3d(0., 1., 0.)
    dirZ = mk.Direction3d(0., 0., 1.)
    hm.SetDirection(dirX, dirY, dirZ)
    hm.SetVolumeFile(f'{file_path}/cardiac.raw', 512, 512, 361)
    hm.SetSpacing(0.351, 0.351, 0.3)
    # hm.SetVolumeFile(f'{file_path}/body.raw', 512, 512, 1559)
    # hm.SetSpacing(0.7422, 0.7422, 1.0)
    tf = {}
    tf[0] = mk.RGBA(0.8, 0, 0, 0)
    tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
    tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
    tf[99] = mk.RGBA(1, 0.8, 1, 1)
    ww = 500
    wl = 250
    hm.SetTransferFunc(tf)
    hm.SetColorBackground(mk.RGBA(0., 0., 0.4, 1.0))
    vol1 = hm.GetVolumeArray()
    for i in range(1):
        print(i)
        time.sleep(2)
        vol1 = (np.random.rand(512, 512, 361)*300).astype(np.int16)
        hm.SetVolumeArray(vol1)
    print(vol1.shape)
    vr = hm.GetVRArray(768, 768)
    b64str = hm.GetVRData_pngString(512, 512)
    # print(b64str)

    b64str_mpr = hm.GetPlaneData_pngString(mk.PlaneSagittal)
    # print(b64str)

def test_set_data(): 
    hm = mk.HelloMonkey()
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

    origin = mk.Point3d(itk_img.GetOrigin()[0], itk_img.GetOrigin()[1], itk_img.GetOrigin()[2])
    hm.SetDirection(dirX, dirY, dirZ)
    hm.SetOrigin(origin)
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

    hm.GetOriginData_pngString(250)

    hm.SetCPRLinePatientArray(np.array([
        [
            36.81390382,
            -156.6719971,
            -424.0480042
        ],
        [
            36.81390382,
            -156.6719971,
            -361.5480042
        ]
    ]))

    # b64str = hm.GetVRData_pngString(512, 512)
    # hm.SaveVR2Png('multivol.png', 512, 512)

    hm.GetPlaneData_pngString(mk.PlaneStretchedCPR)


def test_load_nrrd():
    hm = mk.HelloMonkey()
    tf = {}
    tf[0] = mk.RGBA(0.8, 0, 0, 0)
    tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
    tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
    tf[99] = mk.RGBA(1, 0.8, 1, 1)
    ww = 500
    wl = 250
    hm.LoadVolumeFile(f'{file_path}/cardiac.mhd')
    hm.SetVRWWWL(ww, wl)
    hm.SetTransferFunc(tf)

    # hm.SaveVR2Png(f'{file_path}/a.png', 512, 512)

    # vr = hm.GetVRArray(768, 768)
    b64str = hm.GetVRData_pngString()
    print(b64str)

    # hm.GetPlaneData_pngString(mk.PlaneAxial)


def test_instance():

    # info = mk.DeviceInfo()

    tf = {}
    tf[0] = mk.RGBA(0.8, 0, 0, 0)
    tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
    tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
    tf[99] = mk.RGBA(1, 0.8, 1, 1)
    ww = 500
    wl = 250

    hm1 = mk.HelloMonkey()
    hm1.LoadVolumeFile(f'{file_path}/cardiac.mhd')
    hm1.SetVRWWWL(ww, wl)
    hm1.SetTransferFunc(tf)

    hm1.Zoom(0.9)
    print(hm1.GetZoomRatio())

    hm2 = mk.HelloMonkey()
    hm2.LoadVolumeFile(f'{file_path}/body.mhd')
    hm2.SetVRWWWL(ww, wl)
    hm2.SetTransferFunc(tf)

    hm2.Zoom(1.5)

    print(hm1.GetZoomRatio())
    print(hm2.GetZoomRatio())

    for i in range(100):
        b64str = hm1.GetVRData_pngString()
        b64str = hm2.GetVRData_pngString()

    hm1 = {}
    hm2 = {}

    pass


def test_png():
    import cv2
    import copy
    folder = './data/charimgs/mipmap-xhdpi'

    chs = [
        "~", "`", "·", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "-", "+", "=", \
        "[", "]", "{", "}", "\\", "|", ";", "'", ":", "\"", ",", ".", "/", "<", ">", "?", \
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", " ", \
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", \
        "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", \
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", \
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ];

    chs_name = [
        "~", "`", "·", "！", "@", "#", "$", "%", "^", "&", "＊", "（", "）", "_", "-", "+", "=", \
        "[", "]", "{", "}", "＼", "｜", ";", "'", "：", "＂", ",", "。", "／", "＜", "＞", "？", \
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "空格", \
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", \
        "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", \
        "AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ", "KK", "LL", "MM", "NN", \
        "OO", "PP", "QQ", "RR", "SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ"
    ]


    hm = mk.HelloMonkey()
    for i in range(0, 96):
        ch = chs_name[i]
        # print (f"{i}: {ch}")
        img = cv2.imread(f"{folder}/{ch}.png", cv2.IMREAD_UNCHANGED)
        if img is None:
            print (ch)

        img1 = copy.deepcopy(img[:,:,0])
        if ch == 'q' or ch == 'p':
            shp = img1.shape
            img2 = np.zeros(shp)
            n = 2
            img2[n:,:] = img1[:shp[0]-n,:]
            img1 = img2
        elif ch == 'k':
            shp = img1.shape
            n = 2
            img2 = np.zeros((shp[0], shp[1]-n))
            img2[:,:] = img1[:,:shp[1]-n]
            img1 = img2
        elif ch in ['p', 'v', 'w', 'x']:
            shp = img1.shape
            n = 1
            img2 = np.zeros((shp[0], shp[1]-n))
            img2[:,:] = img1[:,:shp[1]-n]
            img1 = img2

        npdatat = img1.swapaxes(1,0)
        
        hm.Transfer2Base64Array(npdatat)
    pass

if __name__ == "__main__":
    test_png()
    # test_objs()
    # test_set_data()
    # test_load_nrrd()
    # test_instance()

    # for i in range(100):
    #     time.sleep(1)
    # pass


