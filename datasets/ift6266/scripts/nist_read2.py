#!/usr/bin/env python

from pylearn.io import filetensor as ft
import pylab, numpy

datapath = '/data/lisa/data/ift6266h10/train_'

f = open(datapath+'data.ft')
d = ft.read(f)

f = open(datapath+'labels.ft')
labels = ft.read(f)

def label2chr(l):
    if l<10:
        return chr(l + ord('0'))
    elif l<36:
        return chr(l-10 + ord('A'))
    else:
        return chr(l-36 + ord('a'))

for i in range(min(d.shape[0],30)):
    pylab.figure()
    pylab.title(label2chr(labels[i]))
    pylab.imshow(d[i].reshape((32,32))/255., pylab.matplotlib.cm.Greys_r, interpolation='nearest')

pylab.show()

