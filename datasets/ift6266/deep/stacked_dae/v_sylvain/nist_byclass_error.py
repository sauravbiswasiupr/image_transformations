#!/usr/bin/python
# coding: utf-8

import ift6266
import pylearn

import numpy 
import theano
import time

import pylearn.version
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import copy
import sys
import os
import os.path

from jobman import DD
import jobman, jobman.sql
from pylearn.io import filetensor

from utils import produit_cartesien_jobs
from copy import copy

from sgd_optimization import SdaSgdOptimizer

#from ift6266.utils.scalar_series import *
from ift6266.utils.seriestables import *
import tables

from ift6266 import datasets
from config import *

'''
Function called by jobman upon launching each job
Its path is the one given when inserting jobs: see EXPERIMENT_PATH
'''
def jobman_entrypoint(state, channel):
    # record mercurial versions of each package
    pylearn.version.record_versions(state,[theano,ift6266,pylearn])
    # TODO: remove this, bad for number of simultaneous requests on DB
    channel.save()

    # For test runs, we don't want to use the whole dataset so
    # reduce it to fewer elements if asked to.
    rtt = None
    if state.has_key('reduce_train_to'):
        rtt = state['reduce_train_to']
    elif REDUCE_TRAIN_TO:
        rtt = REDUCE_TRAIN_TO
 
    n_ins = 32*32
    n_outs = 62 # 10 digits, 26*2 (lower, capitals)
     
    examples_per_epoch = NIST_ALL_TRAIN_SIZE

    PATH = ''
    NIST_BY_CLASS=0



    print "Creating optimizer with state, ", state

    optimizer = SdaSgdOptimizer(dataset=datasets.nist_all(), 
                                    hyperparameters=state, \
                                    n_ins=n_ins, n_outs=n_outs,\
                                    examples_per_epoch=examples_per_epoch, \
                                    max_minibatches=rtt)	


    
    

    if os.path.exists(PATH+'params_finetune_NIST.txt'):
        print ('\n finetune = NIST ')
        optimizer.reload_parameters(PATH+'params_finetune_NIST.txt')
        if NIST_BY_CLASS == 1:
            print "NIST DIGITS"
            optimizer.training_error(datasets.nist_digits(),part=2)
            print "NIST LOWER CASE"
            optimizer.training_error(datasets.nist_lower(),part=2)
            print "NIST UPPER CASE"
            optimizer.training_error(datasets.nist_upper(),part=2)
        else:
            print "P07 valid"
            optimizer.training_error(datasets.nist_P07(),part=1)
            print "PNIST valid"
            optimizer.training_error(datasets.PNIST07(),part=1)
        
    
    if os.path.exists(PATH+'params_finetune_P07.txt'):
        print ('\n finetune = P07 ')
        optimizer.reload_parameters(PATH+'params_finetune_P07.txt')
        if NIST_BY_CLASS == 1:
            print "NIST DIGITS"
            optimizer.training_error(datasets.nist_digits(),part=2)
            print "NIST LOWER CASE"
            optimizer.training_error(datasets.nist_lower(),part=2)
            print "NIST UPPER CASE"
            optimizer.training_error(datasets.nist_upper(),part=2)
        else:
            print "P07 valid"
            optimizer.training_error(datasets.nist_P07(),part=1)
            print "PNIST valid"
            optimizer.training_error(datasets.PNIST07(),part=1)

    
    if os.path.exists(PATH+'params_finetune_NIST_then_P07.txt'):
        print ('\n finetune = NIST then P07')
        optimizer.reload_parameters(PATH+'params_finetune_NIST_then_P07.txt')
        if NIST_BY_CLASS == 1:
            print "NIST DIGITS"
            optimizer.training_error(datasets.nist_digits(),part=2)
            print "NIST LOWER CASE"
            optimizer.training_error(datasets.nist_lower(),part=2)
            print "NIST UPPER CASE"
            optimizer.training_error(datasets.nist_upper(),part=2)
        else:
            print "P07 valid"
            optimizer.training_error(datasets.nist_P07(),part=1)
            print "PNIST valid"
            optimizer.training_error(datasets.PNIST07(),part=1)
    
    if os.path.exists(PATH+'params_finetune_P07_then_NIST.txt'):
        print ('\n finetune = P07 then NIST')
        optimizer.reload_parameters(PATH+'params_finetune_P07_then_NIST.txt')
        if NIST_BY_CLASS == 1:
            print "NIST DIGITS"
            optimizer.training_error(datasets.nist_digits(),part=2)
            print "NIST LOWER CASE"
            optimizer.training_error(datasets.nist_lower(),part=2)
            print "NIST UPPER CASE"
            optimizer.training_error(datasets.nist_upper(),part=2)
        else:
            print "P07 valid"
            optimizer.training_error(datasets.nist_P07(),part=1)
            print "PNIST valid"
            optimizer.training_error(datasets.PNIST07(),part=1)
    
    if os.path.exists(PATH+'params_finetune_PNIST07.txt'):
        print ('\n finetune = PNIST07')
        optimizer.reload_parameters(PATH+'params_finetune_PNIST07.txt')
        if NIST_BY_CLASS == 1:
            print "NIST DIGITS"
            optimizer.training_error(datasets.nist_digits(),part=2)
            print "NIST LOWER CASE"
            optimizer.training_error(datasets.nist_lower(),part=2)
            print "NIST UPPER CASE"
            optimizer.training_error(datasets.nist_upper(),part=2)
        else:
            print "P07 valid"
            optimizer.training_error(datasets.nist_P07(),part=1)
            print "PNIST valid"
            optimizer.training_error(datasets.PNIST07(),part=1)
        
    if os.path.exists(PATH+'params_finetune_PNIST07_then_NIST.txt'):
        print ('\n finetune = PNIST07 then NIST')
        optimizer.reload_parameters(PATH+'params_finetune_PNIST07_then_NIST.txt')
        if NIST_BY_CLASS == 1:
            print "NIST DIGITS"
            optimizer.training_error(datasets.nist_digits(),part=2)
            print "NIST LOWER CASE"
            optimizer.training_error(datasets.nist_lower(),part=2)
            print "NIST UPPER CASE"
            optimizer.training_error(datasets.nist_upper(),part=2)
        else:
            print "P07 valid"
            optimizer.training_error(datasets.nist_P07(),part=1)
            print "PNIST valid"
            optimizer.training_error(datasets.PNIST07(),part=1)
    
    channel.save()

    return channel.COMPLETE



if __name__ == '__main__':


    chanmock = DD({'COMPLETE':0,'save':(lambda:None)})
    jobman_entrypoint(DD(DEFAULT_HP_NIST), chanmock)


