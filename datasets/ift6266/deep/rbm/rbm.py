"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs 
to those without visible-visible and hidden-hidden connections. 
"""

import numpy, time, cPickle, gzip, PIL.Image

import theano
import theano.tensor as T
import os
import pdb
import numpy
import pylab
import time 
import theano.tensor.nnet
import pylearn
#import ift6266
import theano,pylearn.version #,ift6266
from pylearn.io import filetensor as ft
#from ift6266 import datasets

from jobman.tools import DD, flatten
from jobman import sql

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data

class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=32*32, n_hidden=500, \
        W = None, hbias = None, vbias = None, numpy_rng = None, 
        theano_rng = None):
        """ 
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa), 
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing 
        to a shared hidden units bias vector in case RBM is part of a 
        different network

        :param vbias: None for standalone RBMs or a symbolic variable 
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden  = n_hidden


        if W is None : 
           # W is initialized with `initial_W` which is uniformely sampled
           # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
           # the output of uniform if converted using asarray to dtype 
           # theano.config.floatX so that the code is runable on GPU
           initial_W = numpy.asarray( numpy.random.uniform( 
                     low = -numpy.sqrt(6./(n_hidden+n_visible)), 
                     high = numpy.sqrt(6./(n_hidden+n_visible)), 
                     size = (n_visible, n_hidden)), 
                     dtype = theano.config.floatX)
           # theano shared variables for weights and biases
           W = theano.shared(value = initial_W, name = 'W')

        if hbias is None :
           # create shared variable for hidden units bias
           hbias = theano.shared(value = numpy.zeros(n_hidden, 
                               dtype = theano.config.floatX), name='hbias')

        if vbias is None :
            # create shared variable for visible units bias
            vbias = theano.shared(value =numpy.zeros(n_visible, 
                                dtype = theano.config.floatX),name='vbias')

        if numpy_rng is None:    
            # create a number generator 
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None : 
            theano_rng = RandomStreams(numpy_rng.randint(2**30))


        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input if input else T.dmatrix('input')

        self.W          = W
        self.hbias      = hbias
        self.vbias      = vbias
        self.theano_rng = theano_rng
        
        # **** WARNING: It is not a good idea to put things in this list 
        # other than shared variables created in this function.
        self.params     = [self.W, self.hbias, self.vbias]
        self.batch_size = self.input.shape[0]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.sum(T.dot(v_sample, self.vbias))
        hidden_term = T.sum(T.log(1+T.exp(wx_b)))
        return -hidden_term - vbias_term

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of the visibles
        h1_mean = T.nnet.sigmoid(T.dot(v0_sample, self.W) + self.hbias)
        # get a sample of the hiddens given their activation
        h1_sample = self.theano_rng.binomial(size = h1_mean.shape, n = 1, prob = h1_mean)
        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        v1_mean = T.nnet.sigmoid(T.dot(h0_sample, self.W.T) + self.vbias)
        # get a sample of the visible given their activation
        v1_sample = self.theano_rng.binomial(size = v1_mean.shape,n = 1,prob = v1_mean)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling, 
            starting from the hidden state'''
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [v1_mean, v1_sample, h1_mean, h1_sample]
 
    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling, 
            starting from the visible state'''
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [h1_mean, h1_sample, v1_mean, v1_sample]
 
    def cd(self, lr = 0.1, persistent=None, k=1):
        """ 
        This functions implements one step of CD-1 or PCD-1

        :param lr: learning rate used to train the RBM 
        :param persistent: None for CD. For PCD, shared variable containing old state
        of Gibbs chain. This must be a shared variable of size (batch size, number of
        hidden units).

        Returns the updates dictionary. The dictionary contains the update rules for weights
        and biases but also an update of the shared variable used to store the persistent
        chain, if one is used.
        """

        # compute positive phase
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase (the CD-1)
        [nv_mean, nv_sample, nh_mean, nh_sample] = self.gibbs_hvh(chain_start)

        #perform CD-k
        if k-1>0:
            for i in range(k-1):
                [nv_mean, nv_sample, nh_mean, nh_sample] = self.gibbs_hvh(nh_sample)

                

        # determine gradients on RBM parameters
        g_vbias = T.sum( self.input - nv_mean, axis = 0)/self.batch_size
        g_hbias = T.sum( ph_mean    - nh_mean, axis = 0)/self.batch_size
        g_W = T.dot(ph_mean.T, self.input   )/ self.batch_size - \
              T.dot(nh_mean.T, nv_mean      )/ self.batch_size

        gparams = [g_W.T, g_hbias, g_vbias]

        # constructs the update dictionary
        updates = {}
        for gparam, param in zip(gparams, self.params):
           updates[param] = param + gparam * lr

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = T.cast(nh_sample, dtype=theano.config.floatX)
            # pseudo-likelihood is a better proxy for PCD
            cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            cost = self.get_reconstruction_cost(updates, nv_mean)

        return cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name = 'bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.iround(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx]
        # NB: slice(start,stop,step) is the python object used for
        # slicing, e.g. to index matrix x as follows: x[start:stop:step]
        xi_flip = T.setsubtensor(xi, 1-xi[:, bit_i_idx], 
                                 idx_list=(slice(None,None,None),bit_i_idx))

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i}))) 
        cost = self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, nv_mean):
        """Approximation to the reconstruction error"""

        cross_entropy = T.mean(
                T.sum(self.input*T.log(nv_mean) + 
                (1 - self.input)*T.log(1-nv_mean), axis = 1))

        return cross_entropy



def test_rbm(b_size = 20, nhidden = 1000, kk = 1, persistance = 0):
    """
    Demonstrate ***

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM 

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    """

    learning_rate=0.1

#    if data_set==0:
#   	datasets=datasets.nist_all()
#    elif data_set==1:
#        datasets=datasets.nist_P07()
#    elif data_set==2:
#        datasets=datasets.PNIST07()


    data_path = '/data/lisa/data/nist/by_class/'
    f = open(data_path+'all/all_train_data.ft')
    g = open(data_path+'all/all_train_labels.ft')
    h = open(data_path+'all/all_test_data.ft')
    i = open(data_path+'all/all_test_labels.ft')
    
    train_set_x_uint8 = theano.shared(ft.read(f))
    test_set_x_uint8 = theano.shared(ft.read(h))


    train_set_x = T.cast(train_set_x_uint8/255.,theano.config.floatX)
    train_set_y = ft.read(g)
    test_set_x = T.cast(test_set_x_uint8/255.,theano.config.floatX)
    test_set_y = ft.read(i)
    
    f.close()
    g.close()
    i.close()
    h.close()

    #t = len(train_set_x)
    
    # revoir la recuperation des donnees
##    dataset = load_data(dataset)
##
##    train_set_x, train_set_y = datasets[0]
##    test_set_x , test_set_y  = datasets[2]
    training_epochs = 1 # a determiner

    batch_size = b_size    # size of the minibatch

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x_uint8.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images

    rng        = numpy.random.RandomState(123)
    theano_rng = RandomStreams( rng.randint(2**30))

    
    # construct the RBM class
    rbm = RBM( input = x, n_visible=32*32, \
               n_hidden = nhidden, numpy_rng = rng, theano_rng = theano_rng)

    
    # initialize storage fot the persistent chain (state = hidden layer of chain)
    if persistance == 1:
        persistent_chain = theano.shared(numpy.zeros((batch_size, 500)))
        # get the cost and the gradient corresponding to one step of CD
        cost, updates = rbm.cd(lr=learning_rate, persistent=persistent_chain, k= kk)
        
    else:
        # get the cost and the gradient corresponding to one step of CD
        #persistance_chain = None
        cost, updates = rbm.cd(lr=learning_rate, persistent=None, k= kk)
        
    #################################
    #     Training the RBM          #
    #################################
    #os.chdir('~')
    dirname = str(persistance) + '_' + str(nhidden) + '_' + str(b_size) + '_'+ str(kk)
    os.makedirs(dirname)
    os.chdir(dirname)
    print 'yes'
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    print type(batch_size)
    print index.dtype
    train_rbm = theano.function([index], cost,
           updates = updates, 
           givens = { x: train_set_x[index*batch_size:(index+1)*batch_size]})

    print 'yep'
    plotting_time = 0.0
    start_time = time.clock()  
    bufsize = 1000

    # go through training epochs
    costs = []
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
#        for mini_x, mini_y in datasets.train(b_size):
#           mean_cost += [train_rbm(mini_x)]
##           learning_rate = learning_rate - 0.0001
##           learning_rate = learning_rate/(tau+( epoch*batch_index*batch_size))

        #learning_rate = learning_rate/10

        costs.append(numpy.mean(mean_cost))

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix 
        image = PIL.Image.fromarray(tile_raster_images( X = rbm.W.value.T,
                 img_shape = (32,32),tile_shape = (10,10), 
                 tile_spacing=(1,1)))
        image.save('filters_at_epoch_%i.png'%epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time
   
    
  
    #################################
    #     Sampling from the RBM     #
    #################################

    # find out the number of test samples  
    #number_of_test_samples = 100
    number_of_test_samples = test_set_x.value.shape[0]

    #test_set_x, test_y  = datasets.test(100*b_size)
    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - b_size)
    persistent_vis_chain = theano.shared(test_set_x.value[test_idx:test_idx+b_size])

    # define one step of Gibbs sampling (mf = mean-field)
    [hid_mf, hid_sample, vis_mf, vis_sample] =  rbm.gibbs_vhv(persistent_vis_chain)

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
    # run minibatch size chains for gibbs samples (number of negative particles)
    plot_every = b_size

    for idx in xrange(n_samples):

        # do `plot_every` intermediate samplings of which we do not care
        for jdx in  xrange(plot_every):
            vis_mf, vis_sample = sample_fn()

        # construct image
        image = PIL.Image.fromarray(tile_raster_images( 
                                         X          = vis_mf,
                                         img_shape  = (32,32),
                                         tile_shape = (10,10),
                                         tile_spacing = (1,1) ) )
        #print ' ... plotting sample ', idx
        image.save('sample_%i_step_%i.png'%(idx,idx*jdx))

    #save the model
    model = [rbm.W, rbm.vbias, rbm.hbias]
    f = fopen('params.txt', 'w')
    cPickle.dump(model, f, protocol = -1)
    f.close()
    #os.chdir('./..')
    return numpy.mean(costs), pretraining_time*36


def experiment(state, channel):

    (mean_cost, time_execution) = test_rbm(b_size = state.b_size,\
                                           nhidden = state.ndidden,\
                                           kk = state.kk,\
                                           persistance = state.persistance,\
                                           )

    state.mean_costs = mean_costs
    state.time_execution = time_execution
    pylearn.version.record_versions(state,[theano,ift6266,pylearn])
    return channel.COMPLETE

if __name__ == '__main__':
    
    TABLE_NAME='RBM_tapha'

    # DB path...
    test_rbm()
    #db = sql.db('postgres://ift6266h10:f0572cd63b@gershwin/ift6266h10_db/'+ TABLE_NAME)

    #state = DD()
    #for b_size in 50, 75, 100:
    #    state.b_size = b_size
    #    for nhidden in 1000,1250,1500:
    #        state.nhidden = nhidden
    #        for kk in 1,2,3,4:
    #            state.kk = kk
    #            for persistance in 0,1:
    #                state.persistance = persistance
    #                sql.insert_job(rbm.experiment, flatten(state), db)

    
    #db.createView(TABLE_NAME + 'view')