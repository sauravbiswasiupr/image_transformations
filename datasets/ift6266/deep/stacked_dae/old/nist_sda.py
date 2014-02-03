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

from sgd_optimization import SdaSgdOptimizer

from ift6266.utils.scalar_series import *

##############################################################################
# GLOBALS

TEST_CONFIG = False

NIST_ALL_LOCATION = '/data/lisa/data/nist/by_class/all'
JOBDB = 'postgres://ift6266h10@gershwin/ift6266h10_db/fsavard_sda4'
EXPERIMENT_PATH = "ift6266.deep.stacked_dae.nist_sda.jobman_entrypoint"

REDUCE_TRAIN_TO = None
MAX_FINETUNING_EPOCHS = 1000
# number of minibatches before taking means for valid error etc.
REDUCE_EVERY = 1000

if TEST_CONFIG:
    REDUCE_TRAIN_TO = 1000
    MAX_FINETUNING_EPOCHS = 2
    REDUCE_EVERY = 10

# Possible values the hyperparameters can take. These are then
# combined with produit_cartesien_jobs so we get a list of all
# possible combinations, each one resulting in a job inserted
# in the jobman DB.
JOB_VALS = {'pretraining_lr': [0.1, 0.01],#, 0.001],#, 0.0001],
        'pretraining_epochs_per_layer': [10,20],
        'hidden_layers_sizes': [300,800],
        'corruption_levels': [0.1,0.2,0.3],
        'minibatch_size': [20],
        'max_finetuning_epochs':[MAX_FINETUNING_EPOCHS],
        'finetuning_lr':[0.1, 0.01], #0.001 was very bad, so we leave it out
        'num_hidden_layers':[2,3]}

# Just useful for tests... minimal number of epochs
DEFAULT_HP_NIST = DD({'finetuning_lr':0.1,
                       'pretraining_lr':0.1,
                       'pretraining_epochs_per_layer':20,
                       'max_finetuning_epochs':2,
                       'hidden_layers_sizes':800,
                       'corruption_levels':0.2,
                       'minibatch_size':20,
                       #'reduce_train_to':300,
                       'num_hidden_layers':2})

'''
Function called by jobman upon launching each job
Its path is the one given when inserting jobs:
ift6266.deep.stacked_dae.nist_sda.jobman_entrypoint
'''
def jobman_entrypoint(state, channel):
    # record mercurial versions of each package
    pylearn.version.record_versions(state,[theano,ift6266,pylearn])
    channel.save()

    workingdir = os.getcwd()

    print "Will load NIST"

    nist = NIST(minibatch_size=20)

    print "NIST loaded"

    # For test runs, we don't want to use the whole dataset so
    # reduce it to fewer elements if asked to.
    rtt = None
    if state.has_key('reduce_train_to'):
        rtt = state['reduce_train_to']
    elif REDUCE_TRAIN_TO:
        rtt = REDUCE_TRAIN_TO

    if rtt:
        print "Reducing training set to "+str(rtt)+ " examples"
        nist.reduce_train_set(rtt)

    train,valid,test = nist.get_tvt()
    dataset = (train,valid,test)

    n_ins = 32*32
    n_outs = 62 # 10 digits, 26*2 (lower, capitals)

    # b,b',W for each hidden layer 
    # + b,W of last layer (logreg)
    numparams = state.num_hidden_layers * 3 + 2
    series_mux = None
    series_mux = create_series(workingdir, numparams)

    print "Creating optimizer with state, ", state

    optimizer = SdaSgdOptimizer(dataset=dataset, hyperparameters=state, \
                                    n_ins=n_ins, n_outs=n_outs,\
                                    input_divider=255.0, series_mux=series_mux)

    optimizer.pretrain()
    channel.save()

    optimizer.finetune()
    channel.save()

    return channel.COMPLETE

# These Series objects are used to save various statistics
# during the training.
def create_series(basedir, numparams):
    mux = SeriesMultiplexer()

    # comment out series we don't want to save
    mux.add_series(AccumulatorSeries(name="reconstruction_error",
                    reduce_every=REDUCE_EVERY, # every 1000 batches, we take the mean and save
                    mean=True,
                    directory=basedir, flush_every=1))

    mux.add_series(AccumulatorSeries(name="training_error",
                    reduce_every=REDUCE_EVERY, # every 1000 batches, we take the mean and save
                    mean=True,
                    directory=basedir, flush_every=1))

    mux.add_series(BaseSeries(name="validation_error", directory=basedir, flush_every=1))
    mux.add_series(BaseSeries(name="test_error", directory=basedir, flush_every=1))

    mux.add_series(ParamsArrayStats(numparams,name="params",directory=basedir))

    return mux

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

class NIST:
    def __init__(self, minibatch_size, basepath=None, reduce_train_to=None):
        global NIST_ALL_LOCATION

        self.minibatch_size = minibatch_size
        self.basepath = basepath and basepath or NIST_ALL_LOCATION

        self.set_filenames()

        # arrays of 2 elements: .x, .y
        self.train = [None, None]
        self.test = [None, None]

        self.load_train_test()

        self.valid = [[], []]
        self.split_train_valid()
        if reduce_train_to:
            self.reduce_train_set(reduce_train_to)

    def get_tvt(self):
        return self.train, self.valid, self.test

    def set_filenames(self):
        self.train_files = ['all_train_data.ft',
                                'all_train_labels.ft']

        self.test_files = ['all_test_data.ft',
                            'all_test_labels.ft']

    def load_train_test(self):
        self.load_data_labels(self.train_files, self.train)
        self.load_data_labels(self.test_files, self.test)

    def load_data_labels(self, filenames, pair):
        for i, fn in enumerate(filenames):
            f = open(os.path.join(self.basepath, fn))
            pair[i] = filetensor.read(f)
            f.close()

    def reduce_train_set(self, max):
        self.train[0] = self.train[0][:max]
        self.train[1] = self.train[1][:max]

        if max < len(self.test[0]):
            for ar in (self.test, self.valid):
                ar[0] = ar[0][:max]
                ar[1] = ar[1][:max]

    def split_train_valid(self):
        test_len = len(self.test[0])
        
        new_train_x = self.train[0][:-test_len]
        new_train_y = self.train[1][:-test_len]

        self.valid[0] = self.train[0][-test_len:]
        self.valid[1] = self.train[1][-test_len:]

        self.train[0] = new_train_x
        self.train[1] = new_train_y

def test_load_nist():
    print "Will load NIST"

    import time
    t1 = time.time()
    nist = NIST(20)
    t2 = time.time()

    print "NIST loaded. time delta = ", t2-t1

    tr,v,te = nist.get_tvt()

    print "Lenghts: ", len(tr[0]), len(v[0]), len(te[0])

    raw_input("Press any key")

if __name__ == '__main__':

    import sys

    args = sys.argv[1:]

    if len(args) > 0 and args[0] == 'load_nist':
        test_load_nist()

    elif len(args) > 0 and args[0] == 'jobman_insert':
        jobman_insert_nist()

    elif len(args) > 0 and args[0] == 'test_jobman_entrypoint':
        chanmock = DD({'COMPLETE':0,'save':(lambda:None)})
        jobman_entrypoint(DEFAULT_HP_NIST, chanmock)

    else:
        print "Bad arguments"

