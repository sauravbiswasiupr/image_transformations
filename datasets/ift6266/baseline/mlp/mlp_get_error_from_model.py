__docformat__ = 'restructedtext en'

import pdb
import numpy as np
import pylab
import time 
import pylearn
from pylearn.io import filetensor as ft

data_path = '/data/lisa/data/nist/by_class/'
test_data = 'all/all_train_data.ft'
test_labels = 'all/all_train_labels.ft'

def read_test_data(mlp_model):
    
    
    #read the data
    h = open(data_path+test_data)
    i= open(data_path+test_labels)
    raw_test_data = ft.read(h)
    raw_test_labels = ft.read(i)
    i.close()
    h.close()
    
    #read the model chosen
    a=np.load(mlp_model)
    W1=a['W1']
    W2=a['W2']
    b1=a['b1']
    b2=a['b2']
    
    return (W1,b1,W2,b2,raw_test_data,raw_test_labels)
    
    
    

def get_total_test_error(everything):
    
    W1=everything[0]
    b1=everything[1]
    W2=everything[2]
    b2=everything[3]
    test_data=everything[4]
    test_labels=everything[5]
    total_error_count=0
    total_exemple_count=0
    
    nb_error_count=0
    nb_exemple_count=0
    
    char_error_count=0
    char_exemple_count=0
    
    min_error_count=0
    min_exemple_count=0
    
    maj_error_count=0
    maj_exemple_count=0
    
    for i in range(test_labels.size):
        total_exemple_count = total_exemple_count +1
        #get activation for layer 1
        a0=np.dot(np.transpose(W1),np.transpose(test_data[i]/255.0)) + b1
        #add non linear function to layer 1 activation
        a0_out=np.tanh(a0)
        
        #get activation for output layer
        a1= np.dot(np.transpose(W2),a0_out) + b2
        #add non linear function for output activation (softmax)
        a1_exp = np.exp(a1)
        sum_a1=np.sum(a1_exp)
        a1_out=a1_exp/sum_a1
        
        predicted_class=np.argmax(a1_out)
        wanted_class=test_labels[i]
        
        if(predicted_class!=wanted_class):
            total_error_count = total_error_count +1
            
        #get grouped based error
	#with a priori
#        if(wanted_class>9 and wanted_class<35):
#            min_exemple_count=min_exemple_count+1
#            predicted_class=np.argmax(a1_out[10:35])+10
#            if(predicted_class!=wanted_class):
#		min_error_count=min_error_count+1
#        if(wanted_class<10):
#           nb_exemple_count=nb_exemple_count+1
#            predicted_class=np.argmax(a1_out[0:10])
#            if(predicted_class!=wanted_class):
#                nb_error_count=nb_error_count+1
#        if(wanted_class>34):
#            maj_exemple_count=maj_exemple_count+1
#            predicted_class=np.argmax(a1_out[35:])+35
#            if(predicted_class!=wanted_class):
#                maj_error_count=maj_error_count+1
#                
#        if(wanted_class>9):
#            char_exemple_count=char_exemple_count+1
#            predicted_class=np.argmax(a1_out[10:])+10
#            if(predicted_class!=wanted_class):
#                char_error_count=char_error_count+1
		
		
		
	#get grouped based error
	#with no a priori
        if(wanted_class>9 and wanted_class<35):
            min_exemple_count=min_exemple_count+1
            predicted_class=np.argmax(a1_out)
            if(predicted_class!=wanted_class):
		min_error_count=min_error_count+1
        if(wanted_class<10):
            nb_exemple_count=nb_exemple_count+1
            predicted_class=np.argmax(a1_out)
            if(predicted_class!=wanted_class):
                nb_error_count=nb_error_count+1
        if(wanted_class>34):
            maj_exemple_count=maj_exemple_count+1
            predicted_class=np.argmax(a1_out)
            if(predicted_class!=wanted_class):
                maj_error_count=maj_error_count+1
                
        if(wanted_class>9):
            char_exemple_count=char_exemple_count+1
            predicted_class=np.argmax(a1_out)
            if(predicted_class!=wanted_class):
                char_error_count=char_error_count+1
    
    
    #convert to float 
    return ( total_exemple_count,nb_exemple_count,char_exemple_count,min_exemple_count,maj_exemple_count,\
            total_error_count,nb_error_count,char_error_count,min_error_count,maj_error_count,\
            total_error_count*100.0/total_exemple_count*1.0,\
            nb_error_count*100.0/nb_exemple_count*1.0,\
            char_error_count*100.0/char_exemple_count*1.0,\
            min_error_count*100.0/min_exemple_count*1.0,\
            maj_error_count*100.0/maj_exemple_count*1.0)
            
            
    
    
    
    
    
    
    
    
    
    
 