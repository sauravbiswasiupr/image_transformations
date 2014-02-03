#!/usr/bin/env python

'''
creation des ensembles train, valid et test OCR
ensemble valid est trainorig[:80000]
ensemble test est trainorig[80000:160000]
ensemble train est trainorig[160000:]
trainorig est deja shuffled
'''

from pylearn.io import filetensor as ft
import numpy, os

dir1 = '/data/lisa/data/ocr_breuel/filetensor/'
dir2 = "/data/lisa/data/ift6266h10/"

f = open(dir1 + 'unlv-corrected-2010-02-01-shuffled.ft')
d = ft.read(f)
f = open(dir2 + "ocr_valid_data.ft", 'wb')
ft.write(f, d[:80000])
f = open(dir2 + "ocr_test_data.ft", 'wb')
ft.write(f, d[80000:160000])
f = open(dir2 + "ocr_train_data.ft", 'wb')
ft.write(f, d[160000:])

f = open(dir1 + 'unlv-corrected-2010-02-01-labels-shuffled.ft')
d = ft.read(f)
f = open(dir2 + "ocr_valid_labels.ft", 'wb')
ft.write(f, d[:80000])
f = open(dir2 + "ocr_test_labels.ft", 'wb')
ft.write(f, d[80000:160000])
f = open(dir2 + "ocr_train_labels.ft", 'wb')
ft.write(f, d[160000:])

for i in ["train", "valid", "test"]:
    os.chmod(dir2 + "ocr_" + i + "_data.ft", 0744)
    os.chmod(dir2 + "ocr_" + i + "_labels.ft", 0744)



