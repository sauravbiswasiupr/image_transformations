#!/usr/bin/python

import sys
import os, os.path

# do this before importing custom modules
from mnist_config import *

if not (len(sys.argv) > 1 and sys.argv[1] in \
            ('test_jobman_entrypoint', 'run_local')):
    # in those cases don't use isolated code, use dev code
    print "Running experiment isolation code"
    isolate_experiment()

import numpy as N

import theano
import theano.tensor as T

from crbm import CRBM, ConvolutionParams

import pylearn, pylearn.version
from pylearn.datasets import MNIST
from pylearn.io.image_tiling import tile_raster_images

import Image

from pylearn.io.seriestables import *
import tables

import ift6266

import utils

def setup_workdir():
    if not os.path.exists(IMAGE_OUTPUT_DIR):
        os.mkdir(IMAGE_OUTPUT_DIR)
        if not os.path.exists(IMAGE_OUTPUT_DIR):
            print "For some reason mkdir(IMAGE_OUTPUT_DIR) failed!"
            sys.exit(1)
        print "Created image output dir"
    elif os.path.isfile(IMAGE_OUTPUT_DIR):
        print "IMAGE_OUTPUT_DIR is not a directory!"
        sys.exit(1)

#def filename_from_time(suffix):
#    import datetime
#    return str(datetime.datetime.now()) + suffix + ".png"

def jobman_entrypoint(state, channel):
    # record mercurial versions of each package
    pylearn.version.record_versions(state,[theano,ift6266,pylearn])
    channel.save()

    setup_workdir()

    crbm = MnistCrbm(state)
    crbm.train()

    return channel.COMPLETE

class MnistCrbm(object):
    def __init__(self, state):
        self.state = state

        if TEST_CONFIG:
            self.mnist = MNIST.first_1k()
            print "Test config, so loaded MNIST first 1000"
        else:
            self.mnist = MNIST.full()#first_10k()
            print "Loaded MNIST full"

        self.cp = ConvolutionParams( \
                    num_filters=state.num_filters,
                    num_input_planes=1,
                    height_filters=state.filter_size,
                    width_filters=state.filter_size)

        self.image_size = (28,28)

        self.minibatch_size = state.minibatch_size

        self.lr = state.learning_rate
        self.sparsity_lambda = state.sparsity_lambda
        # about 1/num_filters, so only one filter active at a time
        # 40 * 0.05 = ~2 filters active for any given pixel
        self.sparsity_p = state.sparsity_p

        self.crbm = CRBM( \
                    minibatch_size=self.minibatch_size,
                    image_size=self.image_size,
                    conv_params=self.cp,
                    learning_rate=self.lr,
                    sparsity_lambda=self.sparsity_lambda,
                    sparsity_p=self.sparsity_p)
        
        self.num_epochs = state.num_epochs

        self.init_series()
 
    def init_series(self):
        series = {}

        basedir = os.getcwd()

        h5f = tables.openFile(os.path.join(basedir, "series.h5"), "w")

        cd_series_names = self.crbm.cd_return_desc
        series['cd'] = \
            utils.get_accumulator_series_array( \
                h5f, 'cd', cd_series_names,
                REDUCE_EVERY,
                stdout_too=SERIES_STDOUT_TOO)

        sparsity_series_names = self.crbm.sparsity_return_desc
        series['sparsity'] = \
            utils.get_accumulator_series_array( \
                h5f, 'sparsity', sparsity_series_names,
                REDUCE_EVERY,
                stdout_too=SERIES_STDOUT_TOO)

        # so first we create the names for each table, based on 
        # position of each param in the array
        params_stdout = []
        if SERIES_STDOUT_TOO:
            params_stdout = [StdoutAppendTarget()]
        series['params'] = SharedParamsStatisticsWrapper(
                            new_group_name="params",
                            base_group="/",
                            arrays_names=['W','b_h','b_x'],
                            hdf5_file=h5f,
                            index_names=('epoch','minibatch'),
                            other_targets=params_stdout)

        self.series = series

    def train(self):
        num_minibatches = len(self.mnist.train.x) / self.minibatch_size

        for epoch in xrange(self.num_epochs):
            for mb_index in xrange(num_minibatches):
                mb_x = self.mnist.train.x \
                         [mb_index : mb_index+self.minibatch_size]
                mb_x = mb_x.reshape((self.minibatch_size, 1, 28, 28))

                #E_h = crbm.E_h_given_x_func(mb_x)
                #print "Shape of E_h", E_h.shape

                cd_return = self.crbm.CD_step(mb_x)
                sp_return = self.crbm.sparsity_step(mb_x)

                self.series['cd'].append( \
                        (epoch, mb_index), cd_return)
                self.series['sparsity'].append( \
                        (epoch, mb_index), sp_return)

                total_idx = epoch*num_minibatches + mb_index

                if (total_idx+1) % REDUCE_EVERY == 0:
                    self.series['params'].append( \
                        (epoch, mb_index), self.crbm.params)

                if total_idx % VISUALIZE_EVERY == 0:
                    self.visualize_gibbs_result(\
                        mb_x, GIBBS_STEPS_IN_VIZ_CHAIN,
                        "gibbs_chain_"+str(epoch)+"_"+str(mb_index))
                    self.visualize_gibbs_result(mb_x, 1,
                        "gibbs_1_"+str(epoch)+"_"+str(mb_index))
                    self.visualize_filters(
                        "filters_"+str(epoch)+"_"+str(mb_index))
            if TEST_CONFIG:
                # do a single epoch for cluster tests config
                break

        if SAVE_PARAMS:
            utils.save_params(self.crbm.params, "params.pkl")
    
    def visualize_gibbs_result(self, start_x, gibbs_steps, filename):
        # Run minibatch_size chains for gibbs_steps
        x_samples = None
        if not start_x is None:
            x_samples = self.crbm.gibbs_samples_from(start_x, gibbs_steps)
        else:
            x_samples = self.crbm.random_gibbs_samples(gibbs_steps)
        x_samples = x_samples.reshape((self.minibatch_size, 28*28))
 
        tile = tile_raster_images(x_samples, self.image_size,
                    (1, self.minibatch_size), output_pixel_vals=True)

        filepath = os.path.join(IMAGE_OUTPUT_DIR, filename+".png")
        img = Image.fromarray(tile)
        img.save(filepath)

        print "Result of running Gibbs", \
                gibbs_steps, "times outputed to", filepath

    def visualize_filters(self, filename):
        cp = self.cp

        # filter size
        fsz = (cp.height_filters, cp.width_filters)
        tile_shape = (cp.num_filters, cp.num_input_planes)

        filters_flattened = self.crbm.W.value.reshape(
                                (tile_shape[0]*tile_shape[1],
                                fsz[0]*fsz[1]))

        tile = tile_raster_images(filters_flattened, fsz, 
                                    tile_shape, output_pixel_vals=True)

        filepath = os.path.join(IMAGE_OUTPUT_DIR, filename+".png")
        img = Image.fromarray(tile)
        img.save(filepath)

        print "Filters (as images) outputed to", filepath



if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0:
        print "Bad usage"
    elif args[0] == 'jobman_insert':
        utils.jobman_insert_job_vals(JOBDB, EXPERIMENT_PATH, JOB_VALS)
    elif args[0] == 'test_jobman_entrypoint':
        chanmock = DD({'COMPLETE':0,'save':(lambda:None)})
        jobman_entrypoint(DEFAULT_STATE, chanmock)
    elif args[0] == 'run_default':
        setup_workdir()
        mc = MnistCrbm(DEFAULT_STATE)
        mc.train()


