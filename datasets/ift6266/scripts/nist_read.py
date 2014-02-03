#!/usr/bin/env python

# L'execution de "ipython -pylab nist_read.py" est recommande 

# Pour lire les fichiers NIST, qui sont en format filetensor, il vous faut la
# librarie pylearn, disponible en executant:
# hg clone http://hg.assembla.com/pylearn Pylearn
# et en mettant le repertoire Pylearn dans votre PYTHONPATH

from pylearn.io import filetensor as ft
import pylab, numpy

# repertoire qui contient les donnees NIST
# le repertoire suivant va fonctionner si vous etes connecte sur un ordinateur
# du reseau DIRO
datapath = '/data/lisa/data/nist/by_class/'

# le fichier .ft contient chiffres NIST dans un format efficace. Les chiffres
# sont stockes dans une matrice de NxD, ou N est le nombre d'images, est D est
# le nombre de pixels par image (32x32 = 1024). Chaque pixel de l'image est une
# valeur entre 0 et 255, correspondant a un niveau de gris. Les valeurs sont
# stockees comme des uint8, donc des bytes.
f = open(datapath+'digits/digits_train_data.ft')

# Verifier que vous avez assez de memoire pour loader les donnees au complet
# dans le memoire. Sinon, utilisez ft.arraylike, une classe construite
# specialement pour des fichiers qu'on ne souhaite pas loader dans RAM.
d = ft.read(f)

# Affichage d'une image 
pylab.imshow(d[0].reshape((32,32)))
pylab.show()

# NB: N'oubliez pas de diviser les valeurs des pixels par 255. si jamais vous
# utilisez les donnees commes entrees dans un reseaux de neurones et que vous 
# voulez des entres entre 0 et 1.

# digits_train_data.ft contient les images, digits_train_labels.ft contient les
# etiquettes
f = open(datapath+'digits/digits_train_labels.ft')
labels = ft.read(f)
print 'etiquette: ', labels[0]

