import numpy as np
import time
import SimpleITK as sitk
import sys

file_path = f'{sys.path[0]}/../data'

def fake_data():
    w = 512
    h = 512
    d = 360
    npd = np.zeros((d, h, w), dtype=np.uint16)

    for z in range(-180, 180):
        print(z)
        for y in range(-256, 256):
            for x in range(-256, 256):
                l = np.uint16(np.sqrt(x*x + y*y + z*z))
                if l < 180:
                    npd[z+180, y+256, x+256] = 180 - l

    img = sitk.GetImageFromArray(npd)
    sitk.WriteImage(img, f'{file_path}/test.nii')

if __name__ == "__main__":
    fake_data()
