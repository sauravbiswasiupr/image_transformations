#!/usr/bin/python
# coding: utf-8

# Generic SdA optimization loop, adapted from the deeplearning.net tutorial

import numpy 
import theano
import time
import datetime
import theano.tensor as T
import sys
#import pickle
import cPickle

from jobman import DD
import jobman, jobman.sql
from copy import copy

from stacked_dae import SdA

from ift6266.utils.seriestables import *

#For test purpose only
buffersize=1000

default_series = { \
        'reconstruction_error' : DummySeries(),
        'training_error' : DummySeries(),
        'validation_error' : DummySeries(),
        'test_error' : DummySeries(),
        'params' : DummySeries()
        }

def itermax(iter, max):
    for i,it in enumerate(iter):
        if i >= max:
            break
        yield it

class SdaSgdOptimizer:
    def __init__(self, dataset, hyperparameters, n_ins, n_outs,
                    examples_per_epoch, series=default_series, max_minibatches=None):
        self.dataset = dataset
        self.hp = hyperparameters
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.parameters_pre=[]
   
        self.max_minibatches = max_minibatches
        print "SdaSgdOptimizer, max_minibatches =", max_minibatches

        self.ex_per_epoch = examples_per_epoch
        self.mb_per_epoch = examples_per_epoch / self.hp.minibatch_size

        self.series = series

        self.rng = numpy.random.RandomState(1234)

        self.init_classifier()

        sys.stdout.flush()

    def init_classifier(self):
        print "Constructing classifier"

        # we don't want to save arrays in DD objects, so
        # we recreate those arrays here
        nhl = self.hp.num_hidden_layers
        layers_sizes = [self.hp.hidden_layers_sizes] * nhl
        corruption_levels = [self.hp.corruption_levels] * nhl

        # construct the stacked denoising autoencoder class
        self.classifier = SdA( \
                          batch_size = self.hp.minibatch_size, \
                          n_ins= self.n_ins, \
                          hidden_layers_sizes = layers_sizes, \
                          n_outs = self.n_outs, \
                          corruption_levels = corruption_levels,\
                          rng = self.rng,\
                          pretrain_lr = self.hp.pretraining_lr, \
                          finetune_lr = self.hp.finetuning_lr)

        #theano.printing.pydotprint(self.classifier.pretrain_functions[0], "function.graph")

        sys.stdout.flush()

    def train(self):
        self.pretrain(self.dataset)
        self.finetune(self.dataset)

    def pretrain(self,dataset,decrease=0):
        print "STARTING PRETRAINING, time = ", datetime.datetime.now()
        sys.stdout.flush()
        
        un_fichier=int(819200.0/self.hp.minibatch_size) #Number of batches in a P07 file

        start_time = time.clock()  
        
        ########  This is hardcoaded. THe 0.95 parameter is hardcoaded and can be changed at will  ###
        #Set the decreasing rate of the learning rate. We want the final learning rate to
        #be 5% of the original learning rate. The decreasing factor is linear
        decreasing = (decrease*self.hp.pretraining_lr)/float(self.hp.pretraining_epochs_per_layer*800000/self.hp.minibatch_size)
        
        ## Pre-train layer-wise 
        for i in xrange(self.classifier.n_layers):
            # go through pretraining epochs 
            
            #To reset the learning rate to his original value
            learning_rate=self.hp.pretraining_lr
            for epoch in xrange(self.hp.pretraining_epochs_per_layer):
                # go through the training set
                batch_index=0
                count=0
                num_files=0
                for x,y in dataset.train(self.hp.minibatch_size):
                    c = self.classifier.pretrain_functions[i](x,learning_rate)
                    count +=1

                    self.series["reconstruction_error"].append((epoch, batch_index), c)
                    batch_index+=1

                    #If we need to decrease the learning rate for the pretrain
                    if decrease != 0:
                        learning_rate -= decreasing

                    # useful when doing tests
                    if self.max_minibatches and batch_index >= self.max_minibatches:
                        break
                    
                    #When we pass through the data only once (the case with P07)
                    #There is approximately 800*1024=819200 examples per file (1k per example and files are 800M)
                    if self.hp.pretraining_epochs_per_layer == 1 and count%un_fichier == 0:
                        print 'Pre-training layer %i, epoch %d, cost '%(i,num_files),c
                        num_files+=1
                        sys.stdout.flush()
                        self.series['params'].append((num_files,), self.classifier.all_params)
                
                #When NIST is used
                if self.hp.pretraining_epochs_per_layer > 1:        
                    print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),c
                    sys.stdout.flush()

                    self.series['params'].append((epoch,), self.classifier.all_params)
     
        end_time = time.clock()

        print ('Pretraining took %f minutes' %((end_time-start_time)/60.))
        self.hp.update({'pretraining_time': end_time-start_time})
        
        sys.stdout.flush()
        
        #To be able to load them later for tests on finetune
        self.parameters_pre=[copy(x.value) for x in self.classifier.params]
        f = open('params_pretrain.txt', 'w')
        cPickle.dump(self.parameters_pre,f,protocol=-1)
        f.close()


    def finetune(self,dataset,dataset_test,num_finetune,ind_test,special=0,decrease=0,dataset_test2=None):
        
        if special != 0 and special != 1:
            sys.exit('Bad value for variable special. Must be in {0,1}')
        print "STARTING FINETUNING, time = ", datetime.datetime.now()

        minibatch_size = self.hp.minibatch_size
        if ind_test == 0 or ind_test == 20:
            nom_test = "NIST"
            nom_train="P07"
        elif ind_test == 30:
            nom_train = "PNIST07"
            nom_test = "NIST"
            nom_test2 = "P07"
        elif ind_test == 31:
            nom_train = "NIST"
            nom_test = "PNIST07"
            nom_test2 = "P07"
        else:
            nom_test = "P07"
            nom_train = "NIST"


        # create a function to compute the mistakes that are made by the model
        # on the validation set, or testing set
        test_model = \
            theano.function(
                [self.classifier.x,self.classifier.y], self.classifier.errors)
        #         givens = {
        #           self.classifier.x: ensemble_x,
        #           self.classifier.y: ensemble_y]})

        validate_model = \
            theano.function(
                [self.classifier.x,self.classifier.y], self.classifier.errors)
        #        givens = {
        #           self.classifier.x: ,
        #           self.classifier.y: ]})


        # early-stopping parameters
        patience              = 10000 # look as this many examples regardless
        patience_increase     = 2.    # wait this much longer when a new best is 
                                      # found
        improvement_threshold = 0.995 # a relative improvement of this much is 
                                      # considered significant
        validation_frequency  = min(self.mb_per_epoch, patience/2)
                                      # go through this many 
                                      # minibatche before checking the network 
                                      # on the validation set; in this case we 
                                      # check every epoch 
        if self.max_minibatches and validation_frequency > self.max_minibatches:
            validation_frequency = self.max_minibatches / 2

        best_params          = None
        best_validation_loss = float('inf')
        test_score           = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0

        total_mb_index = 0
        minibatch_index = 0
        parameters_finetune=[]
        
        if ind_test == 21 | ind_test == 31:
            learning_rate = self.hp.finetuning_lr / 10.0
        else:
            learning_rate = self.hp.finetuning_lr  #The initial finetune lr


        while (epoch < num_finetune) and (not done_looping):
            epoch = epoch + 1

            for x,y in dataset.train(minibatch_size,bufsize=buffersize):
                minibatch_index += 1
                
                
                if special == 0:
                    cost_ij = self.classifier.finetune(x,y,learning_rate)
                elif special == 1:
                    cost_ij = self.classifier.finetune2(x,y)
                total_mb_index += 1

                self.series["training_error"].append((epoch, minibatch_index), cost_ij)

                if (total_mb_index+1) % validation_frequency == 0: 
                    #minibatch_index += 1
                    #The validation set is always NIST (we want the model to be good on NIST)
                    if ind_test == 0 | ind_test == 20 | ind_test == 30:
                        iter=dataset_test.valid(minibatch_size,bufsize=buffersize)                        
                    else:
                        iter = dataset.valid(minibatch_size,bufsize=buffersize)
                    if self.max_minibatches:
                        iter = itermax(iter, self.max_minibatches)
                    validation_losses = [validate_model(x,y) for x,y in iter]
                    this_validation_loss = numpy.mean(validation_losses)

                    self.series["validation_error"].\
                        append((epoch, minibatch_index), this_validation_loss*100.)

                    print('epoch %i, minibatch %i, validation error on NIST : %f %%' % \
                           (epoch, minibatch_index+1, \
                            this_validation_loss*100.))


                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold :
                            patience = max(patience, total_mb_index * patience_increase)

                        # save best validation score, iteration number and parameters
                        best_validation_loss = this_validation_loss
                        best_iter = total_mb_index
                        parameters_finetune=[copy(x.value) for x in self.classifier.params]

                        # test it on the test set
                        iter = dataset.test(minibatch_size,bufsize=buffersize)
                        if self.max_minibatches:
                            iter = itermax(iter, self.max_minibatches)
                        test_losses = [test_model(x,y) for x,y in iter]
                        test_score = numpy.mean(test_losses)
                        
                        #test it on the second test set
                        iter2 = dataset_test.test(minibatch_size,bufsize=buffersize)
                        if self.max_minibatches:
                            iter2 = itermax(iter2, self.max_minibatches)
                        test_losses2 = [test_model(x,y) for x,y in iter2]
                        test_score2 = numpy.mean(test_losses2)
                        
                        #test it on the third test set if there is one
                        iter3 = dataset_test2.test(minibatch_size, bufsize=buffersize)
                        if self.max_minibatches:
                            iter3 = itermax(iter3, self.max_minibatches)
                        test_losses3 = [test_model(x,y) for x,y in iter3]
                        test_score3 = numpy.mean(test_losses3)

                        self.series["test_error"].\
                            append((epoch, minibatch_index), test_score*100.)

                        print(('     epoch %i, minibatch %i, test error on dataset %s  (train data) of best '
                              'model %f %%') % 
                                     (epoch, minibatch_index+1,nom_train,
                                      test_score*100.))
                                    
                        print(('     epoch %i, minibatch %i, test error on dataset %s of best '
                              'model %f %%') % 
                                     (epoch, minibatch_index+1,nom_test,
                                      test_score2*100.))
                        print(('     epoch %i, minibatch %i, test error on dataset %s of best '
                              'model %f %%') % 
                                     (epoch, minibatch_index+1,nom_test2,
                                      test_score3*100.))
                    
                    if patience <= total_mb_index:
                        done_looping = True
                        break   #to exit the FOR loop
                    
                    sys.stdout.flush()

                # useful when doing tests
                if self.max_minibatches and minibatch_index >= self.max_minibatches:
                    break
            
            if decrease == 1:
                if (ind_test == 21 & epoch % 100 == 0) | ind_test == 20 | ind_test == 30 | (ind_test == 31 & epoch % 100 == 0):
                    learning_rate /= 2 #divide the learning rate by 2 for each new epoch of P07 (or 100 of NIST)
            
            self.series['params'].append((epoch,), self.classifier.all_params)

            if done_looping == True:    #To exit completly the fine-tuning
                break   #to exit the WHILE loop

        end_time = time.clock()
        self.hp.update({'finetuning_time':end_time-start_time,\
                    'best_validation_error':best_validation_loss,\
                    'test_score':test_score,
                    'num_finetuning_epochs':epoch})

        print(('\nOptimization complete with best validation score of %f %%,'
               'with test performance %f %% on dataset %s ') %  
                     (best_validation_loss * 100., test_score*100.,nom_train))
        print(('The test score on the %s dataset is %f')%(nom_test,test_score2*100.))
        print(('The test score on the %s dataset is %f')%(nom_test2,test_score3*100.))
        
        print ('The finetuning ran for %f minutes' % ((end_time-start_time)/60.))
        
        sys.stdout.flush()
        
        #Save a copy of the parameters in a file to be able to get them in the future
        
        if special == 1:    #To keep a track of the value of the parameters
            f = open('params_finetune_stanford.txt', 'w')
            cPickle.dump(parameters_finetune,f,protocol=-1)
            f.close()
        
        elif ind_test == 0 | ind_test == 20:    #To keep a track of the value of the parameters
            f = open('params_finetune_P07.txt', 'w')
            cPickle.dump(parameters_finetune,f,protocol=-1)
            f.close()
               

        elif ind_test== 1:    #For the run with 2 finetunes. It will be faster.
            f = open('params_finetune_NIST.txt', 'w')
            cPickle.dump(parameters_finetune,f,protocol=-1)
            f.close()
        
        elif ind_test== 21:    #To keep a track of the value of the parameters
            f = open('params_finetune_P07_then_NIST.txt', 'w')
            cPickle.dump(parameters_finetune,f,protocol=-1)
            f.close()
        elif ind_test == 30:
            f = open('params_finetune_PNIST07.txt', 'w')
            cPickle.dump(parameters_finetune,f,protocol=-1)
            f.close()
        elif ind_test == 31:
            f = open('params_finetune_PNIST07_then_NIST.txt', 'w')
            cPickle.dump(parameters_finetune,f,protocol=-1)
            f.close()
        

    #Set parameters like they where right after pre-train or finetune
    def reload_parameters(self,which):
        
        #self.parameters_pre=pickle.load('params_pretrain.txt')
        f = open(which)
        self.parameters_pre=cPickle.load(f)
        f.close()
        for idx,x in enumerate(self.parameters_pre):
            if x.dtype=='float64':
                self.classifier.params[idx].value=theano._asarray(copy(x),dtype=theano.config.floatX)
            else:
                self.classifier.params[idx].value=copy(x)

    def training_error(self,dataset,part=0):
        import math
        # create a function to compute the mistakes that are made by the model
        # on the validation set, or testing set
        test_model = \
            theano.function(
                [self.classifier.x,self.classifier.y], self.classifier.errors)
        #train
        if part == 0:      
            iter2 = dataset.train(self.hp.minibatch_size,bufsize=buffersize)
            name = 'train'
        #validation
        if part == 1:
            iter2 = dataset.valid(self.hp.minibatch_size,bufsize=buffersize)
            name = 'validation'
        if part == 2:
            iter2 = dataset.test(self.hp.minibatch_size,bufsize=buffersize)
            name = 'test'
        train_losses2 = [test_model(x,y) for x,y in iter2]
        train_score2 = numpy.mean(train_losses2)
        print 'On the ' + name + 'dataset'
        print(('\t the error is %f')%(train_score2*100.))
        #print len(train_losses2)
        stderr = math.sqrt(train_score2-train_score2**2)/math.sqrt(len(train_losses2)*self.hp.minibatch_size)
        print (('\t the stderr is %f')%(stderr*100.))
    
    #To see the prediction of the model, the real answer and the image to judge    
    def see_error(self, dataset):
        import pylab
        #The function to know the prediction
        test_model = \
            theano.function(
                [self.classifier.x,self.classifier.y], self.classifier.logLayer.y_pred)
        user = []
        nb_total = 0     #total number of exemples seen
        nb_error = 0   #total number of errors
        for x,y in dataset.test(1):
            nb_total += 1
            pred = self.translate(test_model(x,y))
            rep =  self.translate(y)
            error = pred != rep
            print 'prediction: ' + str(pred) +'\t answer: ' + str(rep) + '\t right: ' + str(not(error))
            pylab.imshow(x.reshape((32,32)))
            pylab.draw()
            if error:
                nb_error += 1
                user.append(int(raw_input("1 = The error is normal, 0 = The error is not normal : ")))
                print '\t\t character is hard to distinguish: ' + str(user[-1])
            else:
                time.sleep(3)
        print '\n Over the '+str(nb_total)+' exemples, there is '+str(nb_error)+' errors. \nThe percentage of errors is'+ str(float(nb_error)/float(nb_total))
        print 'The percentage of errors done by the model that an human will also do: ' + str(numpy.mean(user))
        
            
            
            
    #To translate the numeric prediction in character if necessary     
    def translate(self,y):
        
        if y <= 9:
            return y[0]
        elif y == 10:
            return 'A'
        elif y == 11:
            return 'B'
        elif y == 12:
            return 'C'
        elif y == 13:
            return 'D'
        elif y == 14:
            return 'E'
        elif y == 15:
            return 'F'
        elif y == 16:
            return 'G'
        elif y == 17:
            return 'H'
        elif y == 18:
            return 'I'
        elif y == 19:
            return 'J'
        elif y == 20:
            return 'K'
        elif y == 21:
            return 'L'
        elif y == 22:
            return 'M'
        elif y == 23:
            return 'N'
        elif y == 24:
            return 'O'
        elif y == 25:
            return 'P'
        elif y == 26:
            return 'Q'
        elif y == 27:
            return 'R'
        elif y == 28:
            return 'S'
        elif y == 29:
            return 'T'
        elif y == 30:
            return 'U'
        elif y == 31:
            return 'V'
        elif y == 32:
            return 'W'
        elif y == 33:
            return 'X'
        elif y == 34:
            return 'Y'
        elif y == 35:
            return 'Z'
            
        elif y == 36:
            return 'a'
        elif y == 37:
            return 'b'
        elif y == 38:
            return 'c'
        elif y == 39:
            return 'd'
        elif y == 40:
            return 'e'
        elif y == 41:
            return 'f'
        elif y == 42:
            return 'g'
        elif y == 43:
            return 'h'
        elif y == 44:
            return 'i'
        elif y == 45:
            return 'j'
        elif y == 46:
            return 'k'
        elif y == 47:
            return 'l'
        elif y == 48:
            return 'm'
        elif y == 49:
            return 'n'
        elif y == 50:
            return 'o'
        elif y == 51:
            return 'p'
        elif y == 52:
            return 'q'
        elif y == 53:
            return 'r'
        elif y == 54:
            return 's'
        elif y == 55:
            return 't'
        elif y == 56:
            return 'u'
        elif y == 57:
            return 'v'
        elif y == 58:
            return 'w'
        elif y == 59:
            return 'x'
        elif y == 60:
            return 'y'
        elif y == 61:
            return 'z'    




