#import math

#def sin(x):
#    if 2 * x == pi:
#        return 0.99999999
#    else:
#        return None
    
#pi = 3.14

#print(dir(math))

#print(sin(pi/2))
#print(math.sin(math.pi/2))

def sin(x):
    if 2 * x == pi:
        return 0.99999999
    else:
        return None
    
pi = 3.14
print(sin(pi/2))

from math import sin, pi
print(sin(pi/2))

import math as m
    
print(m.sin(m.pi/2))

from math import pi, radians, degrees, sin, cos, tan, asin

ad = 90
ar = radians(ad)
ad = degrees(ar)

print(ad == 90.)
print(ar == pi/2.)
print(sin(ar) / cos(ar) == tan(ar))
print(asin(sin(ar)) == ar)

from math import ceil, floor, trunc

x = 1.4
y = 2.6

print(floor(x), floor(y))
print(floor(-x), floor(-y))
print(ceil(x), ceil(y))
print(ceil(-x), ceil(-y))
print(trunc(x), trunc(y))
print(trunc(-x), trunc(-y))

#import random
#print(dir(random))

from random import random, seed

seed(0)

for i in range(5):
    print(random())

from platform import platform

print(platform())
print(platform(1))
print(platform(0, 1))

from platform import machine
 
print(machine())

from platform import processor

print(processor())

from platform import system

print(system())