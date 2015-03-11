#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

x   = m_load('x32.in')
y1  = m_load('xAc32.in')
#y2  = m_load('ySec32.in')
y3  = m_load('xTAc32.in')
plt.plot(x, y1, 'b')
#plt.plot(x, y2, 'r')
plt.plot(x, y3, 'g')

plt.hold()
plt.savefig("comparativaAc32.png");
