#!/usr/bin/env python

'''
creation des ensembles train, valid et test NIST pur
ensemble test est pris tel quel
ensemble valid est trainorig[:80000]
ensemble train est trainorig[80000:]
trainorig est deja shuffled
'''

from pylearn.io import filetensor as ft
import numpy, os

dir1 = "/data/lisa/data/nist/by_class/all/"
dir2 = "/data/lisa/data/ift6266h10/"

os.system("cp %s %s" % (dir1 + "all_test_data.ft", dir2 + "test_data.ft"))
os.system("cp %s %s" % (dir1 + "all_test_labels.ft", dir2 + "test_labels.ft"))

f = open(dir1 + "/all_train_data.ft")
d = ft.read(f)
f = open(dir2 + "valid_data.ft", 'wb')
ft.write(f, d[:80000])
f = open(dir2 + "train_data.ft", 'wb')
ft.write(f, d[80000:])

f = open(dir1 + "/all_train_labels.ft")
d = ft.read(f)
f = open(dir2 + "valid_labels.ft", 'wb')
ft.write(f, d[:80000])
f = open(dir2 + "train_labels.ft", 'wb')
ft.write(f, d[80000:])

for i in ["train", "valid", "test"]:
    os.chmod(dir2 + i + "_data.ft", 0744)
    os.chmod(dir2 + i + "_labels.ft", 0744)



