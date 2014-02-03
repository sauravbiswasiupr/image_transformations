#!/usr/bin/env python



from pylearn.io import filetensor as ft
import copy
import pygame
import time
import numpy as N

from ttf2jpg import ttf2jpg

#from gimpfu import *


from PoivreSel import PoivreSel
from thick import Thick
from BruitGauss import BruitGauss
from DistorsionGauss import DistorsionGauss
from PermutPixel import PermutPixel
from gimp_script import GIMP1
from Rature import Rature
from contrast import Contrast
from local_elastic_distortions import LocalElasticDistorter
from slant import Slant
from Occlusion import Occlusion
from add_background_image import AddBackground
from affine_transform import AffineTransformation

###---------------------order of transformation module
MODULE_INSTANCES = [Slant(),Thick(),AffineTransformation(),LocalElasticDistorter(),GIMP1(False)]

###---------------------complexity associated to each of them
complexity = 0.7
#complexity = [0.5]*len(MODULE_INSTANCES)
#complexity = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
n=100

def createimage(path,d):
    for i in range(n):
        screen.fill(0)
        a=d[i,:]
        off1=4*32
        off2=0
        for u in range(n):
            b=N.asarray(N.reshape(a,(32,32)))
            c=N.asarray([N.reshape(a*255.0,(32,32))]*3).T
            new=pygame.surfarray.make_surface(c)
            new=pygame.transform.scale2x(new)
            new=pygame.transform.scale2x(new)
            #new.set_palette(anglcolorpalette)
            screen.blit(new,(0,0))
            exemple.blit(new,(0,0))
            
            offset = 4*32
            offset2 = 0
            ct = 0
            ctmp =  N.random.rand()*complexity
            print u
            for j in MODULE_INSTANCES:
                #max dilation
                #ctmp = N.random.rand()*complexity[ct]
                ctmp = N.random.rand()*complexity 
                #print j.get_settings_names(), j.regenerate_parameters(ctmp)
                th=j.regenerate_parameters(ctmp)
                
                b=j.transform_image(b)
                c=N.asarray([b*255]*3).T
                new=pygame.surfarray.make_surface(c)
                new=pygame.transform.scale2x(new)
                new=pygame.transform.scale2x(new)
                if u==0:
                    #new.set_palette(anglcolorpalette)
                    screen.blit(new,(offset,offset2))
                    font = pygame.font.SysFont('liberationserif',18)
                    text = font.render('%s '%(int(ctmp*100.0)/100.0) + j.__module__,0,(255,255,255),(0,0,0))
                    #if  j.__module__ == 'Rature':
                    #     text = font.render('%s,%s'%(th[-1],int(ctmp*100.0)/100.0) + j.__module__,0,(255,255,255),(0,0,0))
                    screen.blit(text,(offset,offset2+4*32))
                    if ct == len(MODULE_INSTANCES)/2-1:
                        offset = 0
                        offset2 = 4*32+20
                    else:
                        offset += 4*32
                    ct+=1
            exemple.blit(new,(off1,off2))
            if off1 != 9*4*32:
                off1+=4*32
            else:
                off1=0
                off2+=4*32
        pygame.image.save(exemple,path+'/perimages/%s.PNG'%i)
        pygame.image.save(screen,path+'/exemples/%s.PNG'%i)
 



nbmodule = len(MODULE_INSTANCES)

pygame.surfarray.use_arraytype('numpy')

#pygame.display.init()
screen = pygame.Surface((4*(nbmodule+1)/2*32,2*(4*32+20)),depth=32)
exemple = pygame.Surface((N.ceil(N.sqrt(n))*4*32,N.ceil(N.sqrt(n))*4*32),depth=32)

anglcolorpalette=[(x,x,x) for x in xrange(0,256)]
#pygame.Surface.set_palette(anglcolorpalette)
#screen.set_palette(anglcolorpalette)

pygame.font.init()

d = N.zeros((n,1024))

#datapath = '/data/lisa/data/ocr_breuel/filetensor/unlv-corrected-2010-02-01-shuffled.ft'
#f = open(datapath)
#d = ft.read(f)
#d = d[0:n,:]/255.0
#createimage('/u/glorotxa/transf/OCR',d)



datapath = '/data/lisa/data/nist/by_class/'
f = open(datapath+'digits_reshuffled/digits_reshuffled_train_data.ft')
d = ft.read(f)
d = d[0:n,:]/255.0
createimage('/u/glorotxa/transf/NIST_digits',d)



datapath = '/data/lisa/data/nist/by_class/'
f = open(datapath+'upper/upper_train_data.ft')
d = ft.read(f)
d = d[0:n,:]/255.0
createimage('/u/glorotxa/transf/NIST_upper',d)

#from Facade import *

#for i in range(n):
    #d[i,:]=N.asarray(N.reshape(generateCaptcha(0.8,0),(1,1024))/255.0,dtype='float32')

#createimage('/u/glorotxa/transf/capcha',d)


#for i in range(n):
    #myttf2jpg = ttf2jpg()
    #d[i,:]=N.reshape(myttf2jpg.generate_image()[0],(1,1024))
#createimage('/u/glorotxa/transf/fonts',d)

datapath = '/data/lisa/data/nist/by_class/'
f = open(datapath+'lower/lower_train_data.ft')
d = ft.read(f)
d = d[0:n,:]/255.0
createimage('/u/glorotxa/transf/NIST_lower',d)


#pygame.display.quit()
