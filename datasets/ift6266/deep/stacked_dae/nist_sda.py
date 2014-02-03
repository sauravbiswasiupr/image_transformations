#!/usr/bin/python
# coding: utf-8

# Must be imported first
from config import *

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

from utils import produit_cartesien_jobs, jobs_from_reinsert_list

from sgd_optimization import SdaSgdOptimizer

#from ift6266.utils.scalar_series import *
from ift6266.utils.seriestables import *
import tables

from ift6266 import datasets

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
    if rtt:
        examples_per_epoch = rtt

    series = create_series(state.num_hidden_layers)

    print "Creating optimizer with state, ", state

    dataset = None
    if rtt:
        dataset = datasets.nist_all(maxsize=rtt)
    else:
        dataset = datasets.nist_all()

    optimizer = SdaSgdOptimizer(dataset=dataset, 
                                    hyperparameters=state, \
                                    n_ins=n_ins, n_outs=n_outs,\
                                    examples_per_epoch=examples_per_epoch, \
                                    series=series,
                                    save_params=SAVE_PARAMS)

    optimizer.pretrain(dataset)
    channel.save()

    optimizer.finetune(dataset)
    channel.save()

    return channel.COMPLETE

# These Series objects are used to save various statistics
# during the training.
def create_series(num_hidden_layers):

    # Replace series we don't want to save with DummySeries, e.g.
    # series['training_error'] = DummySeries()

    series = {}

    basedir = os.getcwd()

    h5f = tables.openFile(os.path.join(basedir, "series.h5"), "w")

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

def jobman_REinsert_nist():
    jobs = jobs_from_reinsert_list(REINSERT_COLS, REINSERT_JOB_VALS)

    db = jobman.sql.db(JOBDB)
    for job in jobs:
        job.update({jobman.sql.EXPERIMENT: EXPERIMENT_PATH})
        jobman.sql.insert_dict(job, db)

    print "reinserted"



if __name__ == '__main__':

    args = sys.argv[1:]

    #if len(args) > 0 and args[0] == 'load_nist':
    #    test_load_nist()

    if len(args) > 0 and args[0] == 'jobman_insert':
        jobman_insert_nist()

    if len(args) > 0 and args[0] == 'reinsert':
        jobman_REinsert_nist()

    elif len(args) > 0 and args[0] == 'test_jobman_entrypoint':
        chanmock = DD({'COMPLETE':0,'save':(lambda:None)})
        jobman_entrypoint(DEFAULT_HP_NIST, chanmock)

    else:
        print "Bad arguments"

