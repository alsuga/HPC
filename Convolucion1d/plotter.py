#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }

plt.title('Grafica de aceleracion', fontdict=font)

x   = m_load('x.in')
#y1  = m_load('ySec.in')
y1  = m_load('yAcB.in')
y2  = m_load('yAcC.in')
y3  = m_load('yAcT.in')
plt.plot(x, y1, 'r', label="Paralelo Basico")
plt.plot(x, y2, 'g', label="Paralelo Cache")
plt.plot(x, y3, 'b', label="Paralelo Cache Tiling")

plt.xlabel('datos', fontdict=font)
plt.ylabel('aceleracion', fontdict=font)

plt.legend(loc='best')
plt.hold()
plt.savefig("aceleracion.png");
