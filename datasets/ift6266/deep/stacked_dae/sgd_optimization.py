#!/usr/bin/python
# coding: utf-8

# Generic SdA optimization loop, adapted from the deeplearning.net tutorial

from __future__ import with_statement

import numpy 
import theano
import time
import datetime
import theano.tensor as T
import sys

from jobman import DD
import jobman, jobman.sql

from stacked_dae import SdA

from ift6266.utils.seriestables import *

default_series = { \
        'reconstruction_error' : DummySeries(),
        'training_error' : DummySeries(),
        'validation_error' : DummySeries(),
        'test_error' : DummySeries(),
        'params' : DummySeries()
        }

class SdaSgdOptimizer:
    def __init__(self, dataset, hyperparameters, n_ins, n_outs,
                    examples_per_epoch, series=default_series, 
                    save_params=False):
        self.dataset = dataset
        self.hp = hyperparameters
        self.n_ins = n_ins
        self.n_outs = n_outs

        self.save_params = save_params

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

    def pretrain(self,dataset):
        print "STARTING PRETRAINING, time = ", datetime.datetime.now()
        sys.stdout.flush()

        start_time = time.clock()  
        ## Pre-train layer-wise 
        for i in xrange(self.classifier.n_layers):
            # go through pretraining epochs 
            for epoch in xrange(self.hp.pretraining_epochs_per_layer):
                # go through the training set
                batch_index=0
                for x,y in dataset.train(self.hp.minibatch_size):
                    c = self.classifier.pretrain_functions[i](x)

                    self.series["reconstruction_error"].append((epoch, batch_index), c)
                    batch_index+=1

                    #if batch_index % 100 == 0:
                    #    print "100 batches"

                print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),c
                sys.stdout.flush()

                self.series['params'].append((epoch,), self.classifier.all_params)
     
        end_time = time.clock()

        print ('Pretraining took %f minutes' %((end_time-start_time)/60.))
        self.hp.update({'pretraining_time': end_time-start_time})

        sys.stdout.flush()

    def finetune(self,dataset):
        print "STARTING FINETUNING, time = ", datetime.datetime.now()

        minibatch_size = self.hp.minibatch_size

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

        best_params          = None
        best_validation_loss = float('inf')
        test_score           = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0

        total_mb_index = 0

        while (epoch < self.hp.max_finetuning_epochs) and (not done_looping):
            epoch = epoch + 1
            minibatch_index = -1
            for x,y in dataset.train(minibatch_size):
                minibatch_index += 1
                cost_ij = self.classifier.finetune(x,y)
                total_mb_index += 1

                self.series["training_error"].append((epoch, minibatch_index), cost_ij)

                if (total_mb_index+1) % validation_frequency == 0: 
                    
                    iter = dataset.valid(minibatch_size)
                    validation_losses = [validate_model(x,y) for x,y in iter]
                    this_validation_loss = numpy.mean(validation_losses)

                    self.series["validation_error"].\
                        append((epoch, minibatch_index), this_validation_loss*100.)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                           (epoch, minibatch_index+1, self.mb_per_epoch, \
                            this_validation_loss*100.))


                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold :
                            patience = max(patience, total_mb_index * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = total_mb_index

                        # test it on the test set
                        iter = dataset.test(minibatch_size)
                        test_losses = [test_model(x,y) for x,y in iter]
                        test_score = numpy.mean(test_losses)

                        self.series["test_error"].\
                            append((epoch, minibatch_index), test_score*100.)

                        print(('     epoch %i, minibatch %i/%i, test error of best '
                              'model %f %%') % 
                                     (epoch, minibatch_index+1, self.mb_per_epoch,
                                      test_score*100.))

                    sys.stdout.flush()

            self.series['params'].append((epoch,), self.classifier.all_params)

            if patience <= total_mb_index:
                done_looping = True
                break

        end_time = time.clock()
        self.hp.update({'finetuning_time':end_time-start_time,\
                    'best_validation_error':best_validation_loss,\
                    'test_score':test_score,
                    'num_finetuning_epochs':epoch})

        if self.save_params:
            save_params(self.classifier.all_params, "weights.dat")

        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %  
                     (best_validation_loss * 100., test_score*100.))
        print ('The finetuning ran for %f minutes' % ((end_time-start_time)/60.))



def save_params(all_params, filename):
    import pickle
    with open(filename, 'wb') as f:
        values = [p.value for p in all_params]

        # -1 for HIGHEST_PROTOCOL
        pickle.dump(values, f, -1)

