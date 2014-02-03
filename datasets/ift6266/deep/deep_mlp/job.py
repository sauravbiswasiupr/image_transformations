#!/usr/bin/env python
# coding: utf-8

'''
Launching

jobman sqlschedules postgres://ift6266h10@gershwin/ift6266h10_sandbox_db/mlp_dumi mlp_jobman.experiment mlp_jobman.conf
'n_hidden={{500,1000,2000}}'
'n_hidden_layers={{2,3}}'
'train_on={{NIST,NISTP,P07}}'
'train_subset={{DIGITS_ONLY,ALL}}'
'learning_rate_log10={{-1.,-2.,-3.}}'

in mlp_jobman.conf:
rng_seed=1234
L1_reg=0.0
L2_reg=0.0
n_epochs=10
minibatch_size=20
'''

import os, sys, copy, operator, time
import theano
import theano.tensor as T
import numpy
from mlp import MLP
from ift6266 import datasets
from pylearn.io.seriestables import *
import tables
from jobman.tools import DD

N_INPUTS = 32*32
REDUCE_EVERY = 250

TEST_RUN = False

TEST_HP = DD({'n_hidden':200,
            'n_hidden_layers': 2,
            'train_on':'NIST',
            'train_subset':'ALL',
            'learning_rate_log10':-2,
            'rng_seed':1234,
            'L1_reg':0.0,
            'L2_reg':0.0,
            'n_epochs':2,
            'minibatch_size':20})

###########################################
# digits datasets
# nist_digits is already in NIST_PATH and in ift6266.datasets
# NOTE: for these datasets the test and valid sets are wrong
#   (don't correspond to the training set... they're just placeholders)

from ift6266.datasets.defs import NIST_PATH, DATA_PATH
TRANSFORMED_DIGITS_PATH = '/data/lisatmp/ift6266h10/data/transformed_digits'

P07_digits = FTDataSet(\
                     train_data = [os.path.join(TRANSFORMED_DIGITS_PATH,\
                                     'data/P07_train'+str(i)+'_data.ft')\
                                        for i in range(0, 100)],
                     train_lbl = [os.path.join(TRANSFORMED_DIGITS_PATH,\
                                     'data/P07_train'+str(i)+'_labels.ft')\
                                        for i in range(0,100)],
                     test_data = [os.path.join(DATA_PATH,'data/P07_test_data.ft')],
                     test_lbl = [os.path.join(DATA_PATH,'data/P07_test_labels.ft')],
                     valid_data = [os.path.join(DATA_PATH,'data/P07_valid_data.ft')],
                     valid_lbl = [os.path.join(DATA_PATH,'data/P07_valid_labels.ft')],
                     indtype=theano.config.floatX, inscale=255., maxsize=None)
             
#Added PNIST
PNIST07_digits = FTDataSet(train_data = [os.path.join(TRANSFORMED_DIGITS_PATH,\
                                            'PNIST07_train'+str(i)+'_data.ft')\
                                                for i in range(0,100)],
                     train_lbl = [os.path.join(TRANSFORMED_DIGITS_PATH,\
                                            'PNIST07_train'+str(i)+'_labels.ft')\
                                                for i in range(0,100)],
                     test_data = [os.path.join(DATA_PATH,'data/PNIST07_test_data.ft')],
                     test_lbl = [os.path.join(DATA_PATH,'data/PNIST07_test_labels.ft')],
                     valid_data = [os.path.join(DATA_PATH,'data/PNIST07_valid_data.ft')],
                     valid_lbl = [os.path.join(DATA_PATH,'data/PNIST07_valid_labels.ft')],
                     indtype=theano.config.floatX, inscale=255., maxsize=None)


# building valid_test_datasets
# - on veut des dataset_obj pour les 3 datasets
#       - donc juste à bâtir FTDataset(train=nimportequoi, test, valid=pNIST etc.)
# - on veut dans l'array mettre des pointeurs vers la fonction either test ou valid
#        donc PAS dataset_obj, mais dataset_obj.train (sans les parenthèses)
def build_test_valid_sets():
    nist_ds = datasets.nist_all()
    pnist_ds = datasets.PNIST07()
    p07_ds = datasets.nist_P07()

    test_valid_fns = [nist_ds.test, nist_ds.valid,
                    pnist_ds.test, pnist_ds.valid,
                    p07_ds.test, p07_ds.valid]

    test_valid_names = ["nist_all__test", "nist_all__valid",
                        "NISTP__test", "NISTP__valid",
                        "P07__test", "P07__valid"]

    return test_valid_fns, test_valid_names

def add_error_series(series, error_name, hdf5_file,
                    index_names=('minibatch_idx',), use_accumulator=False,
                    reduce_every=250):
    # train
    series_base = ErrorSeries(error_name=error_name,
                    table_name=error_name,
                    hdf5_file=hdf5_file,
                    index_names=index_names)

    if use_accumulator:
        series[error_name] = \
                    AccumulatorSeriesWrapper(base_series=series_base,
                        reduce_every=reduce_every)
    else:
        series[error_name] = series_base

TEST_VALID_FNS,TEST_VALID_NAMES = None, None
def compute_and_save_errors(state, mlp, series, hdf5_file, minibatch_idx):
    global TEST_VALID_FNS,TEST_VALID_NAMES

    TEST_VALID_FNS,TEST_VALID_NAMES = build_test_valid_sets()

    # if the training is on digits only, then there'll be a 100%
    # error on digits in the valid/test set... just ignore them
    
    test_fn = theano.function([mlp.input], mlp.logRegressionLayer.y_pred)

    test_batch_size = 100
    for test_ds_fn,test_ds_name in zip(TEST_VALID_FNS,TEST_VALID_NAMES):
        # reset error counts for every test/valid set
        # note: float
        total_errors = total_digit_errors = \
                total_uppercase_errors = total_lowercase_errors = 0.

        total_all = total_lowercase = total_uppercase = total_digit = 0

        for mb_x,mb_y in test_ds_fn(test_batch_size):
            digit_mask = mb_y < 10
            uppercase_mask = mb_y >= 36
            lowercase_mask = numpy.ones((len(mb_x),)) \
                                    - digit_mask - uppercase_mask

            total_all += len(mb_x)
            total_digit += sum(digit_mask)
            total_uppercase += sum(uppercase_mask)
            total_lowercase += sum(lowercase_mask)

            predictions = test_fn(mb_x)

            all_errors = (mb_y != predictions)
            total_errors += sum(all_errors)

            if len(all_errors) != len(digit_mask):
                print "size all", all_errors.shape, " digit", digit_mask.shape
            total_digit_errors += sum(numpy.multiply(all_errors, digit_mask))
            total_uppercase_errors += sum(numpy.multiply(all_errors, uppercase_mask))
            total_lowercase_errors += sum(numpy.multiply(all_errors, lowercase_mask))

        four_errors = [float(total_errors) / total_all,
                        float(total_digit_errors) / total_digit, 
                        float(total_lowercase_errors) / total_lowercase, 
                        float(total_uppercase_errors) / total_uppercase]

        four_errors_names = ["all", "digits", "lower", "upper"]

        # record stats per set
        print "Errors on", test_ds_name, ",".join(four_errors_names),\
                ":", ",".join([str(e) for e in four_errors])

        # now in the state
        for err, errname in zip(four_errors, four_errors_names):
            error_full_name = 'error__'+test_ds_name+'_'+errname
            min_name = 'min_'+error_full_name
            minpos_name = 'minpos_'+error_full_name

            if state.has_key(min_name):
                if state[min_name] > err:
                    state[min_name] = err
                    state[minpos_name] = pos_str
            else:
                # also create the series
                add_error_series(series, error_full_name, hdf5_file,
                            index_names=('minibatch_idx',))
                state[min_name] = err
                state[minpos_name] = minibatch_idx 

            state[minpos_name] = pos_str
            series[error_full_name].append((minibatch_idx,), err)

def jobman_entrypoint(state, channel):
    global TEST_RUN
    minibatch_size = state.minibatch_size

    print_every = 100000
    COMPUTE_ERROR_EVERY = 10**7 / minibatch_size # compute error every 10 million examples
    if TEST_RUN:
        print_every = 100
        COMPUTE_ERROR_EVERY = 1000 / minibatch_size

    print "entrypoint, state is"
    print state

    ######################
    # select dataset and dataset subset, plus adjust epoch num to make number
    # of examples seen independent of dataset
    # exemple: pour le cas DIGITS_ONLY, il faut changer le nombre d'époques
    # et pour le cas NIST pur (pas de transformations), il faut multiplier par 100
    # en partant car on a pas les variations

    # compute this in terms of the P07 dataset size (=80M)
    MINIBATCHES_TO_SEE = state.n_epochs * 8 * (10**6) / minibatch_size

    if state.train_on == 'NIST' and state.train_subset == 'ALL':
        dataset_obj = datasets.nist_all()
    elif state.train_on == 'NIST' and state.train_subset == 'DIGITS_ONLY':
        dataset_obj = datasets.nist_digits()
    elif state.train_on == 'NISTP' and state.train_subset == 'ALL':
        dataset_obj = datasets.PNIST07()
    elif state.train_on == 'NISTP' and state.train_subset == 'DIGITS_ONLY':
        dataset_obj = PNIST07_digits
    elif state.train_on == 'P07' and state.train_subset == 'ALL':
        dataset_obj = datasets.nist_P07()
    elif state.train_on == 'P07' and state.train_subset == 'DIGITS_ONLY':
        dataset_obj = datasets.P07_digits

    dataset = dataset_obj
    
    if state.train_subset == 'ALL':
        n_classes = 62
    elif state.train_subset == 'DIGITS_ONLY':
        n_classes = 10
    else:
        raise NotImplementedError()

    ###############################
    # construct model

    print "constructing model..."
    x     = T.matrix('x')
    y     = T.ivector('y')

    rng = numpy.random.RandomState(state.rng_seed)

    # construct the MLP class
    model = MLP(rng = rng, input=x, n_in=N_INPUTS,
                        n_hidden_layers = state.n_hidden_layers,
                        n_hidden = state.n_hidden, n_out=n_classes)


    # cost and training fn
    cost = T.mean(model.negative_log_likelihood(y)) \
                 + state.L1_reg * model.L1 \
                 + state.L2_reg * model.L2_sqr 

    print "L1, L2: ", state.L1_reg, state.L2_reg

    gradient_nll_wrt_params = []
    for param in model.params:
        gparam = T.grad(cost, param)
        gradient_nll_wrt_params.append(gparam)

    learning_rate = 10**float(state.learning_rate_log10)
    print "Learning rate", learning_rate

    train_updates = {}
    for param, gparam in zip(model.params, gradient_nll_wrt_params):
        train_updates[param] = param - learning_rate * gparam

    train_fn = theano.function([x,y], cost, updates=train_updates)

    #######################
    # create series
    basedir = os.getcwd()

    h5f = tables.openFile(os.path.join(basedir, "series.h5"), "w")

    series = {}
    add_error_series(series, "training_error", h5f,
                    index_names=('minibatch_idx',), use_accumulator=True,
                    reduce_every=REDUCE_EVERY)

    ##########################
    # training loop

    start_time = time.clock()

    print "begin training..."
    print "will train for", MINIBATCHES_TO_SEE, "examples"

    mb_idx = 0

    while(mb_idx*minibatch_size<nb_max_exemples):

        last_costs = []

        for mb_x, mb_y in dataset.train(minibatch_size):
            if TEST_RUN and mb_idx > 1000:
                break
                
            last_cost = train_fn(mb_x, mb_y)
            series["training_error"].append((mb_idx,), last_cost)

            last_costs.append(last_cost)
            if (len(last_costs)+1) % print_every == 0:
                print "Mean over last", print_every, "minibatches: ", numpy.mean(last_costs)
                last_costs = []

            if (mb_idx+1) % COMPUTE_ERROR_EVERY == 0:
                # compute errors
                print "computing errors on all datasets..."
                print "Time since training began: ", (time.clock()-start_time)/60., "minutes"
                compute_and_save_errors(state, model, series, h5f, mb_idx)

        channel.save()

        sys.stdout.flush()

    end_time = time.clock()

    print "-"*80
    print "Finished. Training took", (end_time-start_time)/60., "minutes"
    print state

def run_test():
    global TEST_RUN
    from fsml.job_management import mock_channel
    TEST_RUN = True
    jobman_entrypoint(TEST_HP, mock_channel)

if __name__ == '__main__':
    run_test()

