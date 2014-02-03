#!/usr/bin/python
# coding: utf-8

import numpy 
import theano
import time
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import copy

from utils import update_locals

# taken from LeDeepNet/daa.py
# has a special case when taking log(0) (defined =0)
# modified to not take the mean anymore
from theano.tensor.xlogx import xlogx, xlogy0
# it's target*log(output)
def binary_cross_entropy(target, output, sum_axis=1):
    XE = xlogy0(target, output) + xlogy0((1 - target), (1 - output))
    return -T.sum(XE, axis=sum_axis)

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out) 
        self.W = theano.shared( value=numpy.zeros((n_in,n_out),
                                            dtype = theano.config.floatX) )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared( value=numpy.zeros((n_out,), 
                                            dtype = theano.config.floatX) )
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)
        
        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)

        # list of parameters for this layer
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
       return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self, y):
        # check if y has same dimension of y_pred 
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))

        # check if y is of the correct datatype        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class SigmoidalLayer(object):
    def __init__(self, rng, input, n_in, n_out):
        self.input = input

        W_values = numpy.asarray( rng.uniform( \
              low = -numpy.sqrt(6./(n_in+n_out)), \
              high = numpy.sqrt(6./(n_in+n_out)), \
              size = (n_in, n_out)), dtype = theano.config.floatX)
        self.W = theano.shared(value = W_values)

        b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
        self.b = theano.shared(value= b_values)

        self.output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]



class dA(object):
  def __init__(self, n_visible= 784, n_hidden= 500, corruption_level = 0.1,\
               input = None, shared_W = None, shared_b = None):
    self.n_visible = n_visible
    self.n_hidden  = n_hidden
    
    # create a Theano random generator that gives symbolic random values
    theano_rng = RandomStreams()
    
    if shared_W != None and shared_b != None : 
        self.W = shared_W
        self.b = shared_b
    else:
        # initial values for weights and biases
        # note : W' was written as `W_prime` and b' as `b_prime`

        # W is initialized with `initial_W` which is uniformely sampled
        # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        initial_W = numpy.asarray( numpy.random.uniform( \
              low = -numpy.sqrt(6./(n_hidden+n_visible)), \
              high = numpy.sqrt(6./(n_hidden+n_visible)), \
              size = (n_visible, n_hidden)), dtype = theano.config.floatX)
        initial_b       = numpy.zeros(n_hidden, dtype = theano.config.floatX)
    
    
        # theano shared variables for weights and biases
        self.W       = theano.shared(value = initial_W,       name = "W")
        self.b       = theano.shared(value = initial_b,       name = "b")
    
 
    initial_b_prime= numpy.zeros(n_visible)
    # tied weights, therefore W_prime is W transpose
    self.W_prime = self.W.T 
    self.b_prime = theano.shared(value = initial_b_prime, name = "b'")

    # if no input is given, generate a variable representing the input
    if input == None : 
        # we use a matrix because we expect a minibatch of several examples,
        # each example being a row
        self.x = T.dmatrix(name = 'input') 
    else:
        self.x = input
    # Equation (1)
    # keep 90% of the inputs the same and zero-out randomly selected subset of 10% of the inputs
    # note : first argument of theano.rng.binomial is the shape(size) of 
    #        random numbers that it should produce
    #        second argument is the number of trials 
    #        third argument is the probability of success of any trial
    #
    #        this will produce an array of 0s and 1s where 1 has a 
    #        probability of 1 - ``corruption_level`` and 0 with
    #        ``corruption_level``
    self.tilde_x  = theano_rng.binomial( self.x.shape,  1,  1 - corruption_level, dtype=theano.config.floatX) * self.x
    # Equation (2)
    # note  : y is stored as an attribute of the class so that it can be 
    #         used later when stacking dAs. 
    self.y   = T.nnet.sigmoid(T.dot(self.tilde_x, self.W      ) + self.b)
    # Equation (3)
    #self.z   = T.nnet.sigmoid(T.dot(self.y, self.W_prime) + self.b_prime)
    # Equation (4)
    # note : we sum over the size of a datapoint; if we are using minibatches,
    #        L will  be a vector, with one entry per example in minibatch
    #self.L = - T.sum( self.x*T.log(self.z) + (1-self.x)*T.log(1-self.z), axis=1 ) 
    #self.L = binary_cross_entropy(target=self.x, output=self.z, sum_axis=1)

    # bypassing z to avoid running to log(0)
    z_a = T.dot(self.y, self.W_prime) + self.b_prime
    log_sigmoid = T.log(1.) - T.log(1.+T.exp(-z_a))
    # log(1-sigmoid(z_a))
    log_1_sigmoid = -z_a - T.log(1.+T.exp(-z_a))
    self.L = -T.sum( self.x * (log_sigmoid) \
                    + (1.0-self.x) * (log_1_sigmoid), axis=1 )

    # I added this epsilon to avoid getting log(0) and 1/0 in grad
    # This means conceptually that there'd be no probability of 0, but that
    # doesn't seem to me as important (maybe I'm wrong?).
    #eps = 0.00000001
    #eps_1 = 1-eps
    #self.L = - T.sum( self.x * T.log(eps + eps_1*self.z) \
    #                + (1-self.x)*T.log(eps + eps_1*(1-self.z)), axis=1 )
    # note : L is now a vector, where each element is the cross-entropy cost 
    #        of the reconstruction of the corresponding example of the 
    #        minibatch. We need to compute the average of all these to get 
    #        the cost of the minibatch
    self.cost = T.mean(self.L)

    self.params = [ self.W, self.b, self.b_prime ]


class SdA(object):
    def __init__(self, batch_size, n_ins, 
                 hidden_layers_sizes, n_outs, 
                 corruption_levels, rng, pretrain_lr, finetune_lr):
        # Just to make sure those are not modified somewhere else afterwards
        hidden_layers_sizes = copy.deepcopy(hidden_layers_sizes)
        corruption_levels = copy.deepcopy(corruption_levels)

        update_locals(self, locals())      
 
        self.layers             = []
        self.pretrain_functions = []
        self.params             = []
        # MODIF: added this so we also get the b_primes
        # (not used for finetuning... still using ".params")
        self.all_params         = []
        self.n_layers           = len(hidden_layers_sizes)

        print "Creating SdA with params:"
        print "batch_size", batch_size
        print "hidden_layers_sizes", hidden_layers_sizes
        print "corruption_levels", corruption_levels
        print "n_ins", n_ins
        print "n_outs", n_outs
        print "pretrain_lr", pretrain_lr
        print "finetune_lr", finetune_lr
        print "----"

        if len(hidden_layers_sizes) < 1 :
            raiseException (' You must have at least one hidden layer ')


        # allocate symbolic variables for the data
        #index   = T.lscalar()    # index to a [mini]batch 
        self.x  = T.matrix('x')  # the data is presented as rasterized images
        self.y  = T.ivector('y') # the labels are presented as 1D vector of 
                                 # [int] labels

        for i in xrange( self.n_layers ):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of 
            # the layer below or the input size if we are on the first layer
            if i == 0 :
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0 : 
                layer_input = self.x
            else:
                layer_input = self.layers[-1].output

            layer = SigmoidalLayer(rng, layer_input, input_size, 
                                   hidden_layers_sizes[i] )
            # add the layer to the 
            self.layers += [layer]
            self.params += layer.params
        
            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(input_size, hidden_layers_sizes[i], \
                          corruption_level = corruption_levels[0],\
                          input = layer_input, \
                          shared_W = layer.W, shared_b = layer.b)

            self.all_params += dA_layer.params
        
            # Construct a function that trains this dA
            # compute gradients of layer parameters
            gparams = T.grad(dA_layer.cost, dA_layer.params)
            # compute the list of updates
            updates = {}
            for param, gparam in zip(dA_layer.params, gparams):
                updates[param] = param - gparam * pretrain_lr
            
            # create a function that trains the dA
            update_fn = theano.function([self.x], dA_layer.cost, \
                  updates = updates)#,
            #     givens = { 
            #         self.x : ensemble})
            # collect this function into a list
            #update_fn = theano.function([index], dA_layer.cost, \
            #      updates = updates,
            #      givens = { 
            #         self.x : train_set_x[index*batch_size:(index+1)*batch_size] / self.shared_divider})
            # collect this function into a list
            self.pretrain_functions += [update_fn]

        
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(\
                         input = self.layers[-1].output,\
                         n_in = hidden_layers_sizes[-1], n_out = n_outs)

        self.params += self.logLayer.params
        self.all_params += self.logLayer.params
        # construct a function that implements one step of finetunining

        # compute the cost, defined as the negative log likelihood 
        cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        gparams = T.grad(cost, self.params)
        # compute list of updates
        updates = {}
        for param,gparam in zip(self.params, gparams):
            updates[param] = param - gparam*finetune_lr
            
        self.finetune = theano.function([self.x,self.y], cost, 
                updates = updates)#,
        #        givens = {
        #          self.x : train_set_x[index*batch_size:(index+1)*batch_size]/self.shared_divider,
        #          self.y : train_set_y[index*batch_size:(index+1)*batch_size]} )

        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y

        self.errors = self.logLayer.errors(self.y)

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

