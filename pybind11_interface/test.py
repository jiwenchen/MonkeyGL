import sys
sys.path.append(f'{sys.path[0]}/build')

import pyMonkeyGL as mk
import numpy as np
import time


p1 = mk.DeviceInfo()
a = 10
print (p1.GetCount(a))
print (a)

p2 = mk.RGBA(2, 3, 4, 1)
p2.Print()

def change2(v):
    v["w"] = 10
    v["h"] = 20
    return False

c = {}
r = mk.change(c)
r = change2(c)

input1 = np.array(range(0, 48)).reshape([4, 4, 3])
input2 = np.array(range(50, 50+48)).reshape([4, 4, 3])
var3 = mk.add_arrays_3d(input1,
                                 input2)
print('-'*50)
print('var3', var3)



hm = mk.HelloMonkey()
# hm.SetVolumeFile(f'{sys.path[0]}/../data/512x512x361_0.3510x0.3510x0.3_int16.raw', 512, 512, 361)
# hm.SetAnisotropy(0.351, 0.351, 0.3)
hm.SetVolumeFile(f'{sys.path[0]}/../data/512x512x1559_0.7422x0.7422x1.0_1x0x0x0x1x0x0x0x-1_int16_Ob.raw', 512, 512, 559)
hm.SetAnisotropy(0.7422, 0.7422, 1.0)
tf = {}
tf[450] = mk.RGBA(0.8, 0.4, 0, 0)
tf[1000] = mk.RGBA(1, 0.8, 0, 1)
hm.SetTransferFunc(tf)
vol1 = hm.GetVolumeArray()
vr = hm.GetVRArray(768, 768)


