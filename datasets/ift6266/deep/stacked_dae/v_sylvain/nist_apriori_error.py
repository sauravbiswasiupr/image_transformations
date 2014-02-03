__docformat__ = 'restructedtext en'

import pdb
import numpy 
from numpy import array
import time
import datetime
import pylearn
import copy
import sys
import os
import os.path
from pylearn.io import filetensor as ft
from jobman import DD
from ift6266 import datasets
import cPickle
from copy import copy
import math

from config import *

data_path = '/data/lisa/data/nist/by_class/'
test_data = 'all/all_train_data.ft' 
test_labels = 'all/all_train_labels.ft'
state = DD(DEFAULT_HP_NIST)

#sda_model -> path for the parameters file
#dataset -> the dataset we use for the test
#part -> 0=train, 1=valid, 2=test
#type -> non-linearity type 0=sigmoid, 1=tanh
def test_data(sda_model,dataset,part=2,type=0):
    
    
    f = open(sda_model)
    parameters_pre=cPickle.load(f)
    f.close()
    
    W1 = array(copy(parameters_pre[0]))
    #print 'W1: ' + str(W1.shape)
    b1 = array(copy(parameters_pre[1]))
    #print 'b1: ' + str(b1.shape)
    W2 = array(copy(parameters_pre[2]))
    #print 'W2: ' + str(W2.shape)
    b2 = array(copy(parameters_pre[3]))
    #print 'b2: ' + str(b2.shape)
    W3 = array(copy(parameters_pre[4]))
    #print 'W3: ' + str(W3.shape)
    b3 = array(copy(parameters_pre[5]))
    #print 'b3: ' + str(b3.shape)
    if state['num_hidden_layers'] == 4:
        W4 = array(copy(parameters_pre[6]))
        b4 = array(copy(parameters_pre[7]))
        Wo = array(copy(parameters_pre[8]))
        bo = array(copy(parameters_pre[9]))
    elif state['num_hidden_layers'] == 3:
        Wo = array(copy(parameters_pre[6]))
        #print 'Wo: ' + str(Wo.shape)
        bo = array(copy(parameters_pre[7]))
        #print 'bo: ' + str(bo.shape)
        W4=None
        b4=None
    else:
        print('Number of layers not implemented yet, please do it')
  
    
    total_error_count=0
    total_exemple_count=0
    if part == 0:
        iter = dataset.train(1)
    if part == 1:
        iter = dataset.valid(1)
    if part == 2:
        iter = dataset.test(1)
    for x,y in iter:
        total_exemple_count = total_exemple_count +1
        if type == 1:
            #get output for layer 1
            out1=(numpy.tanh(numpy.dot(x,W1) + b1)+1.0)/2.0
            #get output for layer 2
            out2=(numpy.tanh(numpy.dot(out1,W2) + b2)+1.0)/2.0
            #get output for layer 3
            out3=(numpy.tanh(numpy.dot(out2,W3) + b3)+1.0)/2.0
            #if there is a fourth layer
            if state['num_hidden_layers'] == 4:
                outf = (numpy.tanh(numpy.dot(out3,W4) + b4)+1.0)/2.0
            else:
                outf = array(out3)
        else:
            #get output for layer 1
            out1=1.0/(1.0+numpy.exp(-(numpy.dot(x,W1)+b1)))
            #get output for layer 2
            out2 = 1.0/(1.0+numpy.exp(-(numpy.dot(out1,W2)+b2)))
            #get output for layer 3
            out3 = 1.0/(1.0+numpy.exp(-(numpy.dot(out2,W3)+b3)))
            #if there is a fourth layer
            if state['num_hidden_layers'] == 4:
                outf = 1.0/(1.0+numpy.exp(-(numpy.dot(out3,W4)+b4)))
            else:
                outf = out3
        
        out_act = numpy.dot(outf,Wo)+bo
        
        #add non linear function for output activation (softmax)
        #We can also use sigmoid and results will be the same
        out = numpy.zeros(len(out_act[0]),float)
        a1_exp = numpy.exp(out_act)
        sum_a1=numpy.sum(a1_exp)
        out=a1_exp/sum_a1
##        for i in xrange(len(out_act[0])):
##            out[i]=sigmoid(array(out_act[0,i]))

        #get grouped based error
        #with a priori
        if(y>9 and y<36):
            predicted_class=numpy.argmax(out[0,10:36])+10
            if(predicted_class!=y):
                total_error_count+=1
                
        if(y<10):
            predicted_class=numpy.argmax(out[0,0:10])
            if(predicted_class!=y):
                total_error_count+=1
        if(y>35):
            predicted_class=numpy.argmax(out[0,36:])+36
            if(predicted_class!=y):
                total_error_count+=1
                
    print '\t total exemples count: '+str(total_exemple_count)
    print '\t total error count: '+str(total_error_count)
    print '\t percentage of error: '+str(total_error_count*100.0/total_exemple_count*1.0)+' %'
    

def sigmoid(value):
##    if len(value) > 1:
##        retour = numpy.zeros(len(value),float)
##        for i in xrange(len(value)):
##            retour[i] = (1.0/(1.0+math.exp(-float(value[i]))))
##        return retour
##    else:
##        print len(value)
        return (1.0/(1.0+math.exp(-value)))

if __name__ == '__main__':
    
    args = sys.argv[1:]
    
    if len(args) > 0 and args[0] == 'sigmoid':
        type = 0
    elif len(args) > 0 and args[0] == 'tanh':
        type = 1
    
    part = 2    #0=train, 1=valid, 2=test
    
    PATH = ''   #Can be changed too if model is not in the current drectory
    
    if os.path.exists(PATH+'params_finetune_NIST.txt'):
        start_time = time.clock()  
        print ('\n finetune = NIST ')
        print "NIST DIGITS"
        test_data(PATH+'params_finetune_NIST.txt',datasets.nist_digits(),part=part,type=type)
        print "NIST LOWER CASE"
        test_data(PATH+'params_finetune_NIST.txt',datasets.nist_lower(),part=part,type=type)
        print "NIST UPPER CASE"
        test_data(PATH+'params_finetune_NIST.txt',datasets.nist_upper(),part=part,type=type)
        end_time = time.clock()
        print ('It took %f minutes' %((end_time-start_time)/60.))
        
    
    if os.path.exists(PATH+'params_finetune_P07.txt'):
        start_time = time.clock()  
        print ('\n finetune = P07 ')
        print "NIST DIGITS"
        test_data(PATH+'params_finetune_P07.txt',datasets.nist_digits(),part=part,type=type)
        print "NIST LOWER CASE"
        test_data(PATH+'params_finetune_P07.txt',datasets.nist_lower(),part=part,type=type)
        print "NIST UPPER CASE"
        test_data(PATH+'params_finetune_P07.txt',datasets.nist_upper(),part=part,type=type)
        end_time = time.clock()
        print ('It took %f minutes' %((end_time-start_time)/60.))

    
    if os.path.exists(PATH+'params_finetune_NIST_then_P07.txt'):
        start_time = time.clock()  
        print ('\n finetune = NIST then P07')
        print "NIST DIGITS"
        test_data(PATH+'params_finetune_NIST_then_P07.txt',datasets.nist_digits(),part=part,type=type)
        print "NIST LOWER CASE"
        test_data(PATH+'params_finetune_NIST_then_P07.txt',datasets.nist_lower(),part=part,type=type)
        print "NIST UPPER CASE"
        test_data(PATH+'params_finetune_NIST_then_P07.txt',datasets.nist_upper(),part=part,type=type)
        end_time = time.clock()
        print ('It took %f minutes' %((end_time-start_time)/60.))
    
    if os.path.exists(PATH+'params_finetune_P07_then_NIST.txt'):
        start_time = time.clock()  
        print ('\n finetune = P07 then NIST')
        print "NIST DIGITS"
        test_data(PATH+'params_finetune_P07_then_NIST.txt',datasets.nist_digits(),part=part,type=type)
        print "NIST LOWER CASE"
        test_data(PATH+'params_finetune_P07_then_NIST.txt',datasets.nist_lower(),part=part,type=type)
        print "NIST UPPER CASE"
        test_data(PATH+'params_finetune_P07_then_NIST.txt',datasets.nist_upper(),part=part,type=type)
        end_time = time.clock()
        print ('It took %f minutes' %((end_time-start_time)/60.))
    
    if os.path.exists(PATH+'params_finetune_PNIST07.txt'):
        start_time = time.clock()  
        print ('\n finetune = PNIST07')
        print "NIST DIGITS"
        test_data(PATH+'params_finetune_PNIST07.txt',datasets.nist_digits(),part=part,type=type)
        print "NIST LOWER CASE"
        test_data(PATH+'params_finetune_PNIST07.txt',datasets.nist_lower(),part=part,type=type)
        print "NIST UPPER CASE"
        test_data(PATH+'params_finetune_PNIST07.txt',datasets.nist_upper(),part=part,type=type)
        end_time = time.clock()
        print ('It took %f minutes' %((end_time-start_time)/60.))
        
    if os.path.exists(PATH+'params_finetune_PNIST07_then_NIST.txt'):
        start_time = time.clock()  
        print ('\n finetune = PNIST07 then NIST')
        print "NIST DIGITS"
        test_data(PATH+'params_finetune_PNIST07_then_NIST.txt',datasets.nist_digits(),part=part,type=type)
        print "NIST LOWER CASE"
        test_data(PATH+'params_finetune_PNIST07_then_NIST.txt',datasets.nist_lower(),part=part,type=type)
        print "NIST UPPER CASE"
        test_data(PATH+'params_finetune_PNIST07_then_NIST.txt',datasets.nist_upper(),part=part,type=type)
        end_time = time.clock()
        print ('It took %f minutes' %((end_time-start_time)/60.))
    
    
    
    
    
    
    
    
    
    
 