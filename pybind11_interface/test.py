import sys
sys.path.append(f'{sys.path[0]}/build')

import pyMonkeyGL as m
import numpy as np


p1 = m.DeviceInfo()
a = 10
print (p1.GetCount(a))
print (a)

p2 = m.RGBA(2, 3, 4, 1)
p2.Print()


hi = m.Hi()
hi.SetVolumeFile('../data/512x512x361_0.3510x0.3510x0.3_int16.raw', 512, 512, 361)
hi.SetAnisotropy(0.351, 0.351, 0.3)
