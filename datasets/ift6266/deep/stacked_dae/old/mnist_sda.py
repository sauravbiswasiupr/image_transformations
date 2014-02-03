#!/usr/bin/python
# coding: utf-8

# TODO: This probably doesn't work anymore, adapt to new code in sgd_opt
# Parameterize call to sgd_optimization for MNIST

import numpy 
import theano
import time
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from sgd_optimization import SdaSgdOptimizer
import cPickle, gzip
from jobman import DD

MNIST_LOCATION = '/u/savardf/datasets/mnist.pkl.gz'

def sgd_optimization_mnist(learning_rate=0.1, pretraining_epochs = 2, \
                            pretrain_lr = 0.1, training_epochs = 5, \
                            dataset='mnist.pkl.gz'):
    # Load the dataset 
    f = gzip.open(dataset,'rb')
    # this gives us train, valid, test (each with .x, .y)
    dataset = cPickle.load(f)
    f.close()

    n_ins = 28*28
    n_outs = 10

    hyperparameters = DD({'finetuning_lr':learning_rate,
                       'pretraining_lr':pretrain_lr,
                       'pretraining_epochs_per_layer':pretraining_epochs,
                       'max_finetuning_epochs':training_epochs,
                       'hidden_layers_sizes':[100],
                       'corruption_levels':[0.2],
                       'minibatch_size':20})

    optimizer = SdaSgdOptimizer(dataset, hyperparameters, n_ins, n_outs)
    optimizer.pretrain()
    optimizer.finetune()

if __name__ == '__main__':
    sgd_optimization_mnist(dataset=MNIST_LOCATION)

