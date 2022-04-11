import sys
sys.path.append(f'{sys.path[0]}/build')

import pyMonkeyGL as mk
import numpy as np
import time


e = mk.PlaneNotDefined

p1 = mk.DeviceInfo()
a = 10
print (p1.GetCount(a))
print (a)

p2 = mk.RGBA(2, 3, 4, 1)
p2.Print()

d3 = mk.Direction3d(1.0, 1.0, 1.0)

hm = mk.HelloMonkey()
# hm.SetLogLevel(mk.LogLevelWarn)
hm.SetVolumeFile(f'{sys.path[0]}/../data/cardiac.raw', 512, 512, 361)
hm.SetAnisotropy(0.351, 0.351, 0.3)
# hm.SetVolumeFile(f'{sys.path[0]}/../data/body.raw', 512, 512, 1559)
# hm.SetAnisotropy(0.7422, 0.7422, 1.0)
tf = {}
tf[0] = mk.RGBA(0.8, 0, 0, 0)
tf[10] = mk.RGBA(0.8, 0, 0, 0.3)
tf[40] = mk.RGBA(0.8, 0.8, 0, 0)
tf[99] = mk.RGBA(1, 0.8, 1, 1)
ww = 500
wl = 250
hm.SetTransferFunc(tf)
hm.SetColorBackgroundArray(np.array([0., 0., 0.4, 1.0]))
vol1 = hm.GetVolumeArray()
for i in range(100):
    print(i)
    time.sleep(2)
    vol1 = (np.random.rand(512, 512, 361)*300).astype(np.int16)
    hm.SetVolumeArray(vol1)
print(vol1.shape)
vr = hm.GetVRArray(768, 768)
b64str = hm.GetVRData_pngString(512, 512)
# print(b64str)

b64str_mpr = hm.GetPlaneData_pngString(mk.PlaneAxial)
# print(b64str)

pass


