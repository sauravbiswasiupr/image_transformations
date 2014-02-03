#!/usr/bin/python
# -*- coding: utf-8 -*-
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
        
    if state.has_key('decrease_lr'):
        decrease_lr = state['decrease_lr']
    else :
        decrease_lr = 0
        
    if state.has_key('decrease_lr_pretrain'):
        dec=state['decrease_lr_pretrain']
    else :
        dec=0
 
    n_ins = 32*32

    if state.has_key('subdataset'):
        subdataset_name=state['subdataset']
    else:
        subdataset_name=SUBDATASET_NIST

    #n_outs = 62 # 10 digits, 26*2 (lower, capitals)
    if subdataset_name == "upper":
	n_outs = 26
	subdataset = datasets.nist_upper()
	examples_per_epoch = NIST_UPPER_TRAIN_SIZE
    elif subdataset_name == "lower":
	n_outs = 26
	subdataset = datasets.nist_lower()
	examples_per_epoch = NIST_LOWER_TRAIN_SIZE
    elif subdataset_name == "digits":
	n_outs = 10
	subdataset = datasets.nist_digits()
	examples_per_epoch = NIST_DIGITS_TRAIN_SIZE
    else:
	n_outs = 62
	subdataset = datasets.nist_all()
	examples_per_epoch = NIST_ALL_TRAIN_SIZE
    
    print 'Using subdataset ', subdataset_name

    #To be sure variables will not be only in the if statement
    PATH = ''
    nom_reptrain = ''
    nom_serie = ""
    if state['pretrain_choice'] == 0:
        nom_serie="series_NIST.h5"
    elif state['pretrain_choice'] == 1:
        nom_serie="series_P07.h5"

    series = create_series(state.num_hidden_layers,nom_serie)


    print "Creating optimizer with state, ", state

    optimizer = SdaSgdOptimizer(dataset_name=subdataset_name,\
				    dataset=subdataset,\
                                    hyperparameters=state, \
                                    n_ins=n_ins, n_outs=n_outs,\
                                    examples_per_epoch=examples_per_epoch, \
                                    series=series,
                                    max_minibatches=rtt)

    parameters=[]
    #Number of files of P07 used for pretraining
    nb_file=0

    print('\n\tpretraining with NIST\n')

    optimizer.pretrain(subdataset, decrease = dec) 

    channel.save()
    
    #Set some of the parameters used for the finetuning
    if state.has_key('finetune_set'):
        finetune_choice=state['finetune_set']
    else:
        finetune_choice=FINETUNE_SET
    
    if state.has_key('max_finetuning_epochs'):
        max_finetune_epoch_NIST=state['max_finetuning_epochs']
    else:
        max_finetune_epoch_NIST=MAX_FINETUNING_EPOCHS
    
    if state.has_key('max_finetuning_epochs_P07'):
        max_finetune_epoch_P07=state['max_finetuning_epochs_P07']
    else:
        max_finetune_epoch_P07=max_finetune_epoch_NIST
    
    #Decide how the finetune is done
    
    if finetune_choice == 0:
        print('\n\n\tfinetune with NIST\n\n')
        optimizer.reload_parameters('params_pretrain.txt')
        optimizer.finetune(subdataset,subdataset,max_finetune_epoch_NIST,ind_test=1,decrease=decrease_lr)
        channel.save()
    if finetune_choice == 1:
        print('\n\n\tfinetune with P07\n\n')
        optimizer.reload_parameters('params_pretrain.txt')
        optimizer.finetune(datasets.nist_P07(),datasets.nist_all(),max_finetune_epoch_P07,ind_test=0,decrease=decrease_lr)
        channel.save()
    if finetune_choice == 2:
        print('\n\n\tfinetune with P07 followed by NIST\n\n')
        optimizer.reload_parameters('params_pretrain.txt')
        optimizer.finetune(datasets.nist_P07(),datasets.nist_all(),max_finetune_epoch_P07,ind_test=20,decrease=decrease_lr)
        optimizer.finetune(datasets.nist_all(),datasets.nist_P07(),max_finetune_epoch_NIST,ind_test=21,decrease=decrease_lr)
        channel.save()
    if finetune_choice == 3:
        print('\n\n\tfinetune with NIST only on the logistic regression on top (but validation on P07).\n\
        All hidden units output are input of the logistic regression\n\n')
        optimizer.reload_parameters('params_pretrain.txt')
        optimizer.finetune(datasets.nist_all(),datasets.nist_P07(),max_finetune_epoch_NIST,ind_test=1,special=1,decrease=decrease_lr)
        
        
    if finetune_choice==-1:
        print('\nSERIE OF 4 DIFFERENT FINETUNINGS')
        print('\n\n\tfinetune with NIST\n\n')
        sys.stdout.flush()
        optimizer.reload_parameters('params_pretrain.txt')
        optimizer.finetune(datasets.nist_all(),datasets.nist_P07(),max_finetune_epoch_NIST,ind_test=1,decrease=decrease_lr)
        channel.save()
        print('\n\n\tfinetune with P07\n\n')
        sys.stdout.flush()
        optimizer.reload_parameters('params_pretrain.txt')
        optimizer.finetune(datasets.nist_P07(),datasets.nist_all(),max_finetune_epoch_P07,ind_test=0,decrease=decrease_lr)
        channel.save()
        print('\n\n\tfinetune with P07 (done earlier) followed by NIST (written here)\n\n')
        sys.stdout.flush()
        optimizer.reload_parameters('params_finetune_P07.txt')
        optimizer.finetune(datasets.nist_all(),datasets.nist_P07(),max_finetune_epoch_NIST,ind_test=21,decrease=decrease_lr)
        channel.save()
        print('\n\n\tfinetune with NIST only on the logistic regression on top.\n\
        All hidden units output are input of the logistic regression\n\n')
        sys.stdout.flush()
        optimizer.reload_parameters('params_pretrain.txt')
        optimizer.finetune(datasets.nist_all(),datasets.nist_P07(),max_finetune_epoch_NIST,ind_test=1,special=1,decrease=decrease_lr)
        channel.save()
    
    channel.save()

    return channel.COMPLETE

# These Series objects are used to save various statistics
# during the training.
def create_series(num_hidden_layers, nom_serie):

    # Replace series we don't want to save with DummySeries, e.g.
    # series['training_error'] = DummySeries()

    series = {}

    basedir = os.getcwd()

    h5f = tables.openFile(os.path.join(basedir, nom_serie), "w")

    # reconstruction
    reconstruction_base = \
                ErrorSeries(error_name="reconstruction_error",
                    table_name="reconstruction_error",
                    hdf5_file=h5f,
                    index_names=('epoch','minibatch'),
                    title="Reconstruction error (mean over "+str(REDUCE_EVERY)+" minibatches)")
    series['reconstruction_error'] = \
                AccumulatorSeriesWrapper(base_series=reconstruction_base,
                    reduce_every=REDUCE_EVERY)

    # train
    training_base = \
                ErrorSeries(error_name="training_error",
                    table_name="training_error",
                    hdf5_file=h5f,
                    index_names=('epoch','minibatch'),
                    title="Training error (mean over "+str(REDUCE_EVERY)+" minibatches)")
    series['training_error'] = \
                AccumulatorSeriesWrapper(base_series=training_base,
                    reduce_every=REDUCE_EVERY)

    # valid and test are not accumulated/mean, saved directly
    series['validation_error'] = \
                ErrorSeries(error_name="validation_error",
                    table_name="validation_error",
                    hdf5_file=h5f,
                    index_names=('epoch','minibatch'))

    series['test_error'] = \
                ErrorSeries(error_name="test_error",
                    table_name="test_error",
                    hdf5_file=h5f,
                    index_names=('epoch','minibatch'))

    param_names = []
    for i in range(num_hidden_layers):
        param_names += ['layer%d_W'%i, 'layer%d_b'%i, 'layer%d_bprime'%i]
    param_names += ['logreg_layer_W', 'logreg_layer_b']

    # comment out series we don't want to save
    series['params'] = SharedParamsStatisticsWrapper(
                        new_group_name="params",
                        base_group="/",
                        arrays_names=param_names,
                        hdf5_file=h5f,
                        index_names=('epoch',))

    return series

# Perform insertion into the Postgre DB based on combination
# of hyperparameter values above
# (see comment for produit_cartesien_jobs() to know how it works)
def jobman_insert_nist():
    jobs = produit_cartesien_jobs(JOB_VALS)

    db = jobman.sql.db(JOBDB)
    for job in jobs:
        job.update({jobman.sql.EXPERIMENT: EXPERIMENT_PATH})
        jobman.sql.insert_dict(job, db)

    print "inserted"

if __name__ == '__main__':

    args = sys.argv[1:]

    #if len(args) > 0 and args[0] == 'load_nist':
    #    test_load_nist()

    if len(args) > 0 and args[0] == 'jobman_insert':
        jobman_insert_nist()

    elif len(args) > 0 and args[0] == 'test_jobman_entrypoint':
        chanmock = DD({'COMPLETE':0,'save':(lambda:None)})
        jobman_entrypoint(DD(DEFAULT_HP_NIST), chanmock)

    else:
        print "Bad arguments"

