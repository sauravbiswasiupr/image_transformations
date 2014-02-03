import numpy
import pylab
from pylearn.io import filetensor as ft
from ift6266 import datasets
from ift6266.datasets.ftfile import FTDataSet

import time
import matplotlib.cm as cm


dataset_str = 'P07safe_' #'PNIST07_' # NISTP

base_path = '/data/lisatmp/ift6266h10/data/'+dataset_str
base_output_path = '/data/lisatmp/ift6266h10/data/transformed_digits/'+dataset_str+'train'

fileno = 15

output_data_file = base_output_path+str(fileno)+'_data.ft'
output_labels_file = base_output_path+str(fileno)+'_labels.ft'

dataset_obj = lambda maxsize=None, min_file=0, max_file=100: \
                FTDataSet(train_data = [output_data_file],
                   train_lbl = [output_labels_file],
                   test_data = [base_path+'_test_data.ft'],
                   test_lbl = [base_path+'_test_labels.ft'],
                   valid_data = [base_path+'_valid_data.ft'],
                   valid_lbl = [base_path+'_valid_labels.ft'])
                   # no conversion or scaling... keep data as is
                   #indtype=theano.config.floatX, inscale=255., maxsize=maxsize)

dataset = dataset_obj()
train_ds = dataset.train(1)

for i in range(2983):
    if i < 2900:
        continue
    ex = train_ds.next()
    pylab.ion()
    pylab.clf()
    pylab.imshow(ex[0].reshape(32,32),cmap=cm.gray)
    pylab.draw()
    time.sleep(0.5)

