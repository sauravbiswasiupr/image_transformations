#!/usr/bin/python
# coding: utf-8

import numpy
import theano
import time
import datetime
import theano.tensor as T
import sys
import pickle

from jobman import DD
import jobman, jobman.sql
from copy import copy

from stacked_convolutional_dae_uit import CSdA

from ift6266.utils.seriestables import *

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
def get_conv_shape(kernels,imgshp,batch_size,max_pool_layers):
    # Returns the dimension at the output of the convoluational net
    # and a list of Image and kernel shape for every
    # Convolutional layer
    conv_layers=[]
    init_layer = [ [ kernels[0][0],1,kernels[0][1],kernels[0][2] ],\
                   [ batch_size , 1, imgshp[0], imgshp[1] ],
                    max_pool_layers[0] ]
    conv_layers.append(init_layer)

    conv_n_out = int((32-kernels[0][2]+1)/max_pool_layers[0][0])

    for i in range(1,len(kernels)):
        layer = [ [ kernels[i][0],kernels[i-1][0],kernels[i][1],kernels[i][2] ],\
                  [ batch_size, kernels[i-1][0],conv_n_out,conv_n_out ],
                   max_pool_layers[i] ]
        conv_layers.append(layer)
        conv_n_out = int( (conv_n_out - kernels[i][2]+1)/max_pool_layers[i][0])
    conv_n_out=kernels[-1][0]*conv_n_out**2
    return conv_n_out,conv_layers





class CSdASgdOptimizer:
    def __init__(self, dataset, hyperparameters, n_ins, n_outs,
                    examples_per_epoch, series=default_series, max_minibatches=None):
        self.dataset = dataset
        self.hp = hyperparameters
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.parameters_pre=[]

        self.max_minibatches = max_minibatches
        print "CSdASgdOptimizer, max_minibatches =", max_minibatches

        self.ex_per_epoch = examples_per_epoch
        self.mb_per_epoch = examples_per_epoch / self.hp.minibatch_size

        self.series = series

        self.rng = numpy.random.RandomState(1234)
        self.init_classifier()

        sys.stdout.flush()

    def init_classifier(self):
        print "Constructing classifier"

        n_ins,convlayers = get_conv_shape(self.hp.kernels,self.hp.imgshp,self.hp.minibatch_size,self.hp.max_pool_layers)

        self.classifier = CSdA(n_ins_mlp = n_ins,
                               batch_size = self.hp.minibatch_size,
                               conv_hidden_layers_sizes = convlayers,
                               mlp_hidden_layers_sizes = self.hp.mlp_size, 
                               corruption_levels = self.hp.corruption_levels,
                               rng = self.rng, 
                               n_out = self.n_outs,
                               pretrain_lr = self.hp.pretraining_lr, 
                               finetune_lr = self.hp.finetuning_lr)



        #theano.printing.pydotprint(self.classifier.pretrain_functions[0], "function.graph")

        sys.stdout.flush()

    def train(self):
        self.pretrain(self.dataset)
        self.finetune(self.dataset)

    def pretrain(self,dataset):
        print "STARTING PRETRAINING, time = ", datetime.datetime.now()
        sys.stdout.flush()

        un_fichier=int(819200.0/self.hp.minibatch_size) #Number of batches in a P07 file

        start_time = time.clock()
        ## Pre-train layer-wise
        for i in xrange(self.classifier.n_layers):
            # go through pretraining epochs
            for epoch in xrange(self.hp.pretraining_epochs_per_layer):
                # go through the training set
                batch_index=0
                count=0
                num_files=0
                for x,y in dataset.train(self.hp.minibatch_size):
                    if x.shape[0] != self.hp.minibatch_size:
                        continue
                    c = self.classifier.pretrain_functions[i](x)
                    count +=1

                    self.series["reconstruction_error"].append((epoch, batch_index), c)
                    batch_index+=1

                    #if batch_index % 100 == 0:
                    #    print "100 batches"

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
        pickle.dump(self.parameters_pre,f)
        f.close()
    def finetune(self,dataset,dataset_test,num_finetune,ind_test,special=0,decrease=0):

        if special != 0 and special != 1:
            sys.exit('Bad value for variable special. Must be in {0,1}')
        print "STARTING FINETUNING, time = ", datetime.datetime.now()

        minibatch_size = self.hp.minibatch_size
        if ind_test == 0 or ind_test == 20:
            nom_test = "NIST"
            nom_train="P07"
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
        learning_rate = self.hp.finetuning_lr


        while (epoch < num_finetune) and (not done_looping):
            epoch = epoch + 1

            for x,y in dataset.train(minibatch_size,bufsize=buffersize):

                minibatch_index += 1

                if x.shape[0] != self.hp.minibatch_size:
                    print 'bim'
                    continue

                cost_ij = self.classifier.finetune(x,y)#,learning_rate)
                total_mb_index += 1

                self.series["training_error"].append((epoch, minibatch_index), cost_ij)

                if (total_mb_index+1) % validation_frequency == 0:
                    #minibatch_index += 1
                    #The validation set is always NIST (we want the model to be good on NIST)

                    iter=dataset_test.valid(minibatch_size,bufsize=buffersize)
 

                    if self.max_minibatches:
                        iter = itermax(iter, self.max_minibatches)

                    validation_losses = []

                    for x,y in iter:
                        if x.shape[0] != self.hp.minibatch_size:
                            print 'bim'
                            continue
                        validation_losses.append(validate_model(x,y))

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
                        test_losses = []
                        test_losses2 = []
                        for x,y in iter:
                            if x.shape[0] != self.hp.minibatch_size:
                                print 'bim'
                                continue
                            test_losses.append(test_model(x,y))

                        test_score = numpy.mean(test_losses)

                        #test it on the second test set
                        iter2 = dataset_test.test(minibatch_size,bufsize=buffersize)
                        if self.max_minibatches:
                            iter2 = itermax(iter2, self.max_minibatches)
                        for x,y in iter2:
                            if x.shape[0] != self.hp.minibatch_size:
                                continue
                            test_losses2.append(test_model(x,y))

                        test_score2 = numpy.mean(test_losses2)

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

                    if patience <= total_mb_index:
                        done_looping = True
                        break   #to exit the FOR loop

                    sys.stdout.flush()

                # useful when doing tests
                if self.max_minibatches and minibatch_index >= self.max_minibatches:
                    break

            if decrease == 1:
                learning_rate /= 2 #divide the learning rate by 2 for each new epoch

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

        print ('The finetuning ran for %f minutes' % ((end_time-start_time)/60.))

        sys.stdout.flush()

        #Save a copy of the parameters in a file to be able to get them in the future

        if special == 1:    #To keep a track of the value of the parameters
            f = open('params_finetune_stanford.txt', 'w')
            pickle.dump(parameters_finetune,f)
            f.close()

        elif ind_test == 0 | ind_test == 20:    #To keep a track of the value of the parameters
            f = open('params_finetune_P07.txt', 'w')
            pickle.dump(parameters_finetune,f)
            f.close()


        elif ind_test== 1:    #For the run with 2 finetunes. It will be faster.
            f = open('params_finetune_NIST.txt', 'w')
            pickle.dump(parameters_finetune,f)
            f.close()

        elif ind_test== 21:    #To keep a track of the value of the parameters
            f = open('params_finetune_P07_then_NIST.txt', 'w')
            pickle.dump(parameters_finetune,f)
            f.close()
    #Set parameters like they where right after pre-train or finetune
    def reload_parameters(self,which):

        #self.parameters_pre=pickle.load('params_pretrain.txt')
        f = open(which)
        self.parameters_pre=pickle.load(f)
        f.close()
        for idx,x in enumerate(self.parameters_pre):
            if x.dtype=='float64':
                self.classifier.params[idx].value=theano._asarray(copy(x),dtype=theano.config.floatX)
            else:
                self.classifier.params[idx].value=copy(x)

    def training_error(self,dataset):
        # create a function to compute the mistakes that are made by the model
        # on the validation set, or testing set
        test_model = \
            theano.function(
                [self.classifier.x,self.classifier.y], self.classifier.errors)

        iter2 = dataset.train(self.hp.minibatch_size,bufsize=buffersize)
        train_losses2 = [test_model(x,y) for x,y in iter2]
        train_score2 = numpy.mean(train_losses2)
        print "Training error is: " + str(train_score2)
