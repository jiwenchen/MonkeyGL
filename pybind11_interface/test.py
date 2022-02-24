import pyMonkeyGL as m
import numpy as np


p1 = m.DeviceInfo()
a = 10
print (p1.GetCount(a))
print (a)

p2 = m.RGBA(2, 3, 4, 1)
p2.Print()

print (np.array(m.get_data()))