import sys
import os, os.path

import numpy as N

import theano
import theano.tensor as T

from crbm import CRBM, ConvolutionParams

from pylearn.datasets import MNIST
from pylearn.io.image_tiling import tile_raster_images

import Image

from pylearn.io.seriestables import *
import tables

IMAGE_OUTPUT_DIR = 'img/'

REDUCE_EVERY = 100

def filename_from_time(suffix):
    import datetime
    return str(datetime.datetime.now()) + suffix + ".png"

# Just a shortcut for a common case where we need a few
# related Error (float) series

def get_accumulator_series_array( \
                hdf5_file, group_name, series_names, 
                reduce_every,
                index_names=('epoch','minibatch'),
                stdout_too=True,
                skip_hdf5_append=False):
    all_series = []

    hdf5_file.createGroup('/', group_name)

    other_targets = []
    if stdout_too:
        other_targets = [StdoutAppendTarget()]

    for sn in series_names:
        series_base = \
            ErrorSeries(error_name=sn,
                table_name=sn,
                hdf5_file=hdf5_file,
                hdf5_group='/'+group_name,
                index_names=index_names,
                other_targets=other_targets,
                skip_hdf5_append=skip_hdf5_append)

        all_series.append( \
            AccumulatorSeriesWrapper( \
                    base_series=series_base,
                    reduce_every=reduce_every))

    ret_wrapper = SeriesArrayWrapper(all_series)

    return ret_wrapper

class ExperienceRbm(object):
    def __init__(self):
        self.mnist = MNIST.full()#first_10k()

        
        datasets = load_data(dataset)

        train_set_x, train_set_y = datasets[0]
        test_set_x , test_set_y  = datasets[2]


        batch_size = 100    # size of the minibatch

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.value.shape[0] / batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch 
        x     = T.matrix('x')  # the data is presented as rasterized images

        rng        = numpy.random.RandomState(123)
        theano_rng = RandomStreams( rng.randint(2**30))

        # initialize storage fot the persistent chain (state = hidden layer of chain)
        persistent_chain = theano.shared(numpy.zeros((batch_size, 500)))

        # construct the RBM class
        self.rbm = RBM( input = x, n_visible=28*28, \
                   n_hidden = 500,numpy_rng = rng, theano_rng = theano_rng)

        # get the cost and the gradient corresponding to one step of CD
        

        self.init_series()
 
    def init_series(self):

        series = {}

        basedir = os.getcwd()

        h5f = tables.openFile(os.path.join(basedir, "series.h5"), "w")

        cd_series_names = self.rbm.cd_return_desc
        series['cd'] = \
            get_accumulator_series_array( \
                h5f, 'cd', cd_series_names,
                REDUCE_EVERY,
                stdout_too=True)

       

        # so first we create the names for each table, based on 
        # position of each param in the array
        params_stdout = StdoutAppendTarget("\n------\nParams")
        series['params'] = SharedParamsStatisticsWrapper(
                            new_group_name="params",
                            base_group="/",
                            arrays_names=['W','b_h','b_x'],
                            hdf5_file=h5f,
                            index_names=('epoch','minibatch'),
                            other_targets=[params_stdout])

        self.series = series

    def train(self, persistent, learning_rate):

        training_epochs = 15

        #get the cost and the gradient corresponding to one step of CD
        if persistant:
            persistent_chain = theano.shared(numpy.zeros((batch_size, self.rbm.n_hidden)))
            cost, updates = self.rbm.cd(lr=learning_rate, persistent=persistent_chain)

        else:
            cost, updates = self.rbm.cd(lr=learning_rate)
    
        dirname = 'lr=%.5f'%self.rbm.learning_rate
        os.makedirs(dirname)
        os.chdir(dirname)

        # the purpose of train_rbm is solely to update the RBM parameters
        train_rbm = theano.function([index], cost,
               updates = updates, 
               givens = { x: train_set_x[index*batch_size:(index+1)*batch_size]})

        plotting_time = 0.
        start_time = time.clock()  


        # go through training epochs 
        for epoch in xrange(training_epochs):

            # go through the training set
            mean_cost = []
            for batch_index in xrange(n_train_batches):
               mean_cost += [train_rbm(batch_index)]

    
        pretraining_time = (end_time - start_time)
            

       
    
    def sample_from_rbm(self, gibbs_steps, test_set_x):

        # find out the number of test samples  
        number_of_test_samples = test_set_x.value.shape[0]

        # pick random test examples, with which to initialize the persistent chain
        test_idx = rng.randint(number_of_test_samples-20)
        persistent_vis_chain = theano.shared(test_set_x.value[test_idx:test_idx+20])

        # define one step of Gibbs sampling (mf = mean-field)
        [hid_mf, hid_sample, vis_mf, vis_sample] =  self.rbm.gibbs_vhv(persistent_vis_chain)

        # the sample at the end of the channel is returned by ``gibbs_1`` as 
        # its second output; note that this is computed as a binomial draw, 
        # therefore it is formed of ints (0 and 1) and therefore needs to 
        # be converted to the same dtype as ``persistent_vis_chain``
        vis_sample = T.cast(vis_sample, dtype=theano.config.floatX)

        # construct the function that implements our persistent chain 
        # we generate the "mean field" activations for plotting and the actual samples for
        # reinitializing the state of our persistent chain
        sample_fn = theano.function([], [vis_mf, vis_sample],
                          updates = { persistent_vis_chain:vis_sample})

        # sample the RBM, plotting every `plot_every`-th sample; do this 
        # until you plot at least `n_samples`
        n_samples = 10
        plot_every = 1000

        for idx in xrange(n_samples):

            # do `plot_every` intermediate samplings of which we do not care
            for jdx in  xrange(plot_every):
                vis_mf, vis_sample = sample_fn()

            # construct image
            image = PIL.Image.fromarray(tile_raster_images( 
                                             X          = vis_mf,
                                             img_shape  = (28,28),
                                             tile_shape = (10,10),
                                             tile_spacing = (1,1) ) )
            
            image.save('sample_%i_step_%i.png'%(idx,idx*jdx))    


if __name__ == '__main__':
    mc = ExperienceRbm()
    mc.train()

