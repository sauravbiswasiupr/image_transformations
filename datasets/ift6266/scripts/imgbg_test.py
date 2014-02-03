#!/usr/bin/python

import Image, cPickle

f=open('/Tmp/image_net/filelist.pkl')
image_files = cPickle.load(f)
f.close()

for i in range(len(image_files)):
    filename = '/Tmp/image_net/' + image_files[i]
    try:
        image = Image.open(filename).convert('L')
    except:
        print filename

