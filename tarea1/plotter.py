#! /usr/bin/python
import matplotlib.pyplot as plt
from numpy import *

def m_load(fname) :
    return fromfile(fname, sep='\n')

x  = m_load('x.in')
y  = m_load('xAc.in')
plt.plot(x, y, 'bx')
plt.plot(x, y, 'g')

plt.hold()
plt.savefig("time.png");
