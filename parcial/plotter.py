#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

x   = m_load('x4F.in')
y1  = m_load('xAc4F.in')
#y2  = m_load('ySec4F.in')
y3  = m_load('xTAc4F.in')
plt.plot(x, y1, 'b')
#plt.plot(x, y2, 'r')
plt.plot(x, y3, 'g')

plt.hold()
plt.savefig("comparativaAc4F.png");
