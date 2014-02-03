#!/usr/bin/python
# coding: utf-8

# Generic SdA optimization loop, adapted from the deeplearning.net tutorial

import numpy 
import theano
import time
import datetime
import theano.tensor as T
import sys

from jobman import DD
import jobman, jobman.sql

from stacked_dae import SdA

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    #shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    #shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    #shared_y = T.cast(shared_y, 'int32')
    shared_x = theano.shared(data_x)
    shared_y = theano.shared(data_y)
    return shared_x, shared_y

class DummyMux():
    def append(self, param1, param2):
        pass

class SdaSgdOptimizer:
    def __init__(self, dataset, hyperparameters, n_ins, n_outs, input_divider=1.0, series_mux=None):
        self.dataset = dataset
        self.hp = hyperparameters
        self.n_ins = n_ins
        self.n_outs = n_outs
        self.input_divider = input_divider
   
        if not series_mux:
            series_mux = DummyMux()
            print "No series multiplexer set"
        self.series_mux = series_mux

        self.rng = numpy.random.RandomState(1234)

        self.init_datasets()
        self.init_classifier()

        sys.stdout.flush()
     
    def init_datasets(self):
        print "init_datasets"
        sys.stdout.flush()

        train_set, valid_set, test_set = self.dataset
        self.test_set_x, self.test_set_y = shared_dataset(test_set)
        self.valid_set_x, self.valid_set_y = shared_dataset(valid_set)
        self.train_set_x, self.train_set_y = shared_dataset(train_set)

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.train_set_x.value.shape[0] / self.hp.minibatch_size
        self.n_valid_batches = self.valid_set_x.value.shape[0] / self.hp.minibatch_size
        # remove last batch in case it's incomplete
        self.n_test_batches  = (self.test_set_x.value.shape[0]  / self.hp.minibatch_size) - 1

    def init_classifier(self):
        print "Constructing classifier"

        # we don't want to save arrays in DD objects, so
        # we recreate those arrays here
        nhl = self.hp.num_hidden_layers
        layers_sizes = [self.hp.hidden_layers_sizes] * nhl
        corruption_levels = [self.hp.corruption_levels] * nhl

        # construct the stacked denoising autoencoder class
        self.classifier = SdA( \
                          train_set_x= self.train_set_x, \
                          train_set_y = self.train_set_y,\
                          batch_size = self.hp.minibatch_size, \
                          n_ins= self.n_ins, \
                          hidden_layers_sizes = layers_sizes, \
                          n_outs = self.n_outs, \
                          corruption_levels = corruption_levels,\
                          rng = self.rng,\
                          pretrain_lr = self.hp.pretraining_lr, \
                          finetune_lr = self.hp.finetuning_lr,\
                          input_divider = self.input_divider )

        #theano.printing.pydotprint(self.classifier.pretrain_functions[0], "function.graph")

        sys.stdout.flush()

    def train(self):
        self.pretrain()
        self.finetune()

    def pretrain(self):
        print "STARTING PRETRAINING, time = ", datetime.datetime.now()
        sys.stdout.flush()

        #time_acc_func = 0.0
        #time_acc_total = 0.0

        start_time = time.clock()  
        ## Pre-train layer-wise 
        for i in xrange(self.classifier.n_layers):
            # go through pretraining epochs 
            for epoch in xrange(self.hp.pretraining_epochs_per_layer):
                # go through the training set
                for batch_index in xrange(self.n_train_batches):
                    #t1 = time.clock()
                    c = self.classifier.pretrain_functions[i](batch_index)
                    #t2 = time.clock()

                    #time_acc_func += t2 - t1

                    #if batch_index % 500 == 0:
                    #    print "acc / total", time_acc_func / (t2 - start_time), time_acc_func

                    self.series_mux.append("reconstruction_error", c)
                        
                print 'Pre-training layer %i, epoch %d, cost '%(i,epoch),c
                sys.stdout.flush()

                self.series_mux.append("params", self.classifier.all_params)
     
        end_time = time.clock()

        print ('Pretraining took %f minutes' %((end_time-start_time)/60.))
        self.hp.update({'pretraining_time': end_time-start_time})

        sys.stdout.flush()

    def finetune(self):
        print "STARTING FINETUNING, time = ", datetime.datetime.now()

        index   = T.lscalar()    # index to a [mini]batch 
        minibatch_size = self.hp.minibatch_size

        # create a function to compute the mistakes that are made by the model
        # on the validation set, or testing set
        shared_divider = theano.shared(numpy.asarray(self.input_divider, dtype=theano.config.floatX))
        test_model = theano.function([index], self.classifier.errors,
                 givens = {
                   self.classifier.x: self.test_set_x[index*minibatch_size:(index+1)*minibatch_size] / shared_divider,
                   self.classifier.y: self.test_set_y[index*minibatch_size:(index+1)*minibatch_size]})

        validate_model = theano.function([index], self.classifier.errors,
                givens = {
                   self.classifier.x: self.valid_set_x[index*minibatch_size:(index+1)*minibatch_size] / shared_divider,
                   self.classifier.y: self.valid_set_y[index*minibatch_size:(index+1)*minibatch_size]})


        # early-stopping parameters
        patience              = 10000 # look as this many examples regardless
        patience_increase     = 2.    # wait this much longer when a new best is 
                                      # found
        improvement_threshold = 0.995 # a relative improvement of this much is 
                                      # considered significant
        validation_frequency  = min(self.n_train_batches, patience/2)
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

        while (epoch < self.hp.max_finetuning_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):

                cost_ij = self.classifier.finetune(minibatch_index)
                iter    = epoch * self.n_train_batches + minibatch_index

                self.series_mux.append("training_error", cost_ij)

                if (iter+1) % validation_frequency == 0: 
                    
                    validation_losses = [validate_model(i) for i in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    self.series_mux.append("validation_error", this_validation_loss)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                           (epoch, minibatch_index+1, self.n_train_batches, \
                            this_validation_loss*100.))


                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold :
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(self.n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        self.series_mux.append("test_error", test_score)
                        print(('     epoch %i, minibatch %i/%i, test error of best '
                              'model %f %%') % 
                                     (epoch, minibatch_index+1, self.n_train_batches,
                                      test_score*100.))

                    sys.stdout.flush()

            self.series_mux.append("params", self.classifier.all_params)

            if patience <= iter :
                done_looping = True
                break

        end_time = time.clock()
        self.hp.update({'finetuning_time':end_time-start_time,\
                    'best_validation_error':best_validation_loss,\
                    'test_score':test_score,
                    'num_finetuning_epochs':epoch})

        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %  
                     (best_validation_loss * 100., test_score*100.))
        print ('The finetuning ran for %f minutes' % ((end_time-start_time)/60.))



