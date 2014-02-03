#!/usr/bin/python
# coding: utf-8

'''
Sert a creer un petit jeu de donnees afin de pouvoir avoir des fragments
de lettres pour ajouter bruit d'occlusion

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''

from pylearn.io import filetensor as ft
import pylab
import random as r
from numpy import *

nombre = 20000  #La grandeur de l'echantillon

f = open('all_train_data.ft')  #Le jeu de donnees est en local.  
d = ft.read(f)
f.close()
print len(d)
random.seed(3525)

echantillon=r.sample(xrange(len(d)),nombre)
nouveau=d[0:nombre]
for i in xrange(nombre):
    nouveau[i]=d[echantillon[i]]


f2 = open('echantillon_occlusion.ft', 'w')
ft.write(f2,nouveau)
f2.close()


##Tester si ca a fonctionne
f3 = open('echantillon_occlusion.ft')

d2=ft.read(f3)
pylab.imshow(d2[0].reshape((32,32)))
pylab.show()
f3.close()