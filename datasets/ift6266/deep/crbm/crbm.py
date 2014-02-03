import sys
import os, os.path

import numpy

import theano

USING_GPU = "gpu" in theano.config.device

import theano.tensor as T
from theano.tensor.nnet import conv, sigmoid

if not USING_GPU:
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams

_PRINT_GRAPHS = True

def _init_conv_biases(num_filters, varname, rng=numpy.random):
    b_shp = (num_filters,)
    b = theano.shared( numpy.asarray(
                rng.uniform(low=-.5, high=.5, size=b_shp),
                dtype=theano.config.floatX), name=varname)
    return b

def _init_conv_weights(conv_params, varname, rng=numpy.random):
    cp = conv_params

    # initialize shared variable for weights.
    w_shp = conv_params.as_conv2d_shape_tuple()
    w_bound =  numpy.sqrt(cp.num_input_planes * \
                    cp.height_filters * cp.width_filters)
    W = theano.shared( numpy.asarray(
                rng.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=w_shp),
                dtype=theano.config.floatX), name=varname)

    return W

# Shape of W for conv2d
class ConvolutionParams:
    def __init__(self, num_filters, num_input_planes, height_filters, width_filters):
        self.num_filters = num_filters
        self.num_input_planes = num_input_planes
        self.height_filters = height_filters
        self.width_filters = width_filters

    def as_conv2d_shape_tuple(self):
        cp = self
        return (cp.num_filters, cp.num_input_planes,
                    cp.height_filters, cp.width_filters)

class CRBM:
    def __init__(self, minibatch_size, image_size, conv_params,
                 learning_rate, sparsity_lambda, sparsity_p):
        '''
        Parameters
        ----------
        image_size
            height, width
        '''
        self.minibatch_size = minibatch_size
        self.image_size = image_size
        self.conv_params = conv_params

        '''
        Dimensions:
        0- minibatch
        1- plane/color
        2- y (rows)
        3- x (cols)
        '''
        self.x = T.tensor4('x')
        self.h = T.tensor4('h')

        self.lr = theano.shared(numpy.asarray(learning_rate,
                                    dtype=theano.config.floatX))
        self.sparsity_lambda = \
                theano.shared( \
                    numpy.asarray( \
                        sparsity_lambda, 
                        dtype=theano.config.floatX))
        self.sparsity_p = \
                theano.shared( \
                    numpy.asarray(sparsity_p, \
                            dtype=theano.config.floatX))

        self.numpy_rng = numpy.random.RandomState(1234)

        if not USING_GPU:
            self.theano_rng = RandomStreams(self.numpy_rng.randint(2**30))
        else:
            self.theano_rng = MRG_RandomStreams(234, use_cuda=True)

        self._init_params()
        self._init_functions()

    def _get_visibles_shape(self):
        imsz = self.image_size
        return (self.minibatch_size,
                    self.conv_params.num_input_planes,
                    imsz[0], imsz[1])

    def _get_hiddens_shape(self):
        cp = self.conv_params
        imsz = self.image_size
        wf, hf = cp.height_filters, cp.width_filters
        return (self.minibatch_size, cp.num_filters, 
                    imsz[0] - hf + 1, imsz[1] - wf + 1)

    def _init_params(self):
        cp = self.conv_params

        self.W = _init_conv_weights(cp, 'W')
        self.b_h = _init_conv_biases(cp.num_filters, 'b_h')
        '''
        Lee09 mentions "all visible units share a single bias c"
        but for upper layers it's pretty clear we need one
        per plane, by symmetry
        '''
        self.b_x = _init_conv_biases(cp.num_input_planes, 'b_x')

        self.params = [self.W, self.b_h, self.b_x]

        # flip filters horizontally and vertically
        W_flipped = self.W[:, :, ::-1, ::-1]
        # also have to invert the filters/num_planes
        self.W_tilde = W_flipped.dimshuffle(1,0,2,3)

    '''
    I_up and I_down come from the symbol used in the 
    Lee 2009 CRBM paper
    '''
    def _I_up(self, visibles_mb):
        '''
        output of conv is features maps of size
                image_size - filter_size + 1
        The dimshuffle serves to broadcast b_h so that it 
        corresponds to output planes
        '''
        fshp = self.conv_params.as_conv2d_shape_tuple()
        return conv.conv2d(visibles_mb, self.W,
                    filter_shape=fshp) + \
                    self.b_h.dimshuffle('x',0,'x','x')

    def _I_down(self, hiddens_mb):
        '''
        notice border_mode='full'... we want to get
        back the original size
        so we get feature_map_size + filter_size - 1
        The dimshuffle serves to broadcast b_x so that
        it corresponds to output planes
        '''
        fshp = list(self.conv_params.as_conv2d_shape_tuple())
        # num_filters and num_planes swapped
        fshp[0], fshp[1] = fshp[1], fshp[0]
        return conv.conv2d(hiddens_mb, self.W_tilde, 
                    border_mode='full',filter_shape=tuple(fshp)) + \
                self.b_x.dimshuffle('x',0,'x','x')

    def _mean_free_energy(self, visibles_mb):
        '''
        visibles_mb is mb_size x num_planes x h x w

        we want to match the summed input planes
            (second dimension, first is mb index)
        to respective bias terms for the visibles
        The dimshuffle isn't really necessary, 
            but I put it there for clarity.
        '''
        vbias_term = \
            self.b_x.dimshuffle('x',0) * \
            T.sum(visibles_mb,axis=[2,3])
        # now sum over term per planes, get one free energy 
        # contribution per element of minibatch
        vbias_term = - T.sum(vbias_term, axis=1)

        '''
        Here it's a bit more complex, a few points:
        - The usual free energy, in the fully connected case,
            is a sum over all hiddens.
          We do the same thing here, but each unit has limited
            connectivity and there's weight reuse.
          Therefore we only need to first do the convolutions
            (with I_up) which gives us
          what would normally be the Wx+b_h for each hidden.
            Once we have this,
          we take the log(1+exp(sum for this hidden)) elemwise 
            for each hidden,
          then we sum for all hiddens in one example of the minibatch.
          
        - Notice that we reuse the same b_h everywhere instead of 
            using one b per hidden,
          so the broadcasting for b_h done in I_up is all right.
        
        That sum is over all hiddens, so all filters
             (planes of hiddens), x, and y.
        In the end we get one free energy contribution per
            example of the minibatch.
        '''
        softplused = T.log(1.0+T.exp(self._I_up(visibles_mb)))
        # h_sz = self._get_hiddens_shape()
        # this simplifies the sum
        # num_hiddens = h_sz[1] * h_sz[2] * h_sz[3]
        # reshaped = T.reshape(softplused, 
        #       (self.minibatch_size, num_hiddens))

        # this is because the 0,1,1,1 sum pattern is not 
        # implemented on gpu, but the 1,0,1,1 pattern is
        dimshuffled = softplused.dimshuffle(1,0,2,3)
        xh_and_hbias_term = - T.sum(dimshuffled, axis=[0,2,3])

        '''
        both bias_term and vbias_term end up with one
        contributor to free energy per minibatch
        so we mean over minibatches
        '''
        return T.mean(vbias_term + xh_and_hbias_term)

    def _init_functions(self):
        # propup
        # b_h is broadcasted keeping in mind we want it to
        # correspond to each new plane (corresponding to filters)
        I_up = self._I_up(self.x)
        # expected values for the distributions for each hidden
        E_h_given_x = sigmoid(I_up) 
        # might be needed if we ever want a version where we
        # take expectations instead of samples for CD learning
        self.E_h_given_x_func = theano.function([self.x], E_h_given_x)

        if _PRINT_GRAPHS:
            print "----------------------\nE_h_given_x_func"
            theano.printing.debugprint(self.E_h_given_x_func)

        h_sample_given_x = \
            self.theano_rng.binomial( \
                            size = self._get_hiddens_shape(),
                            n = 1, 
                            p = E_h_given_x, 
                            dtype = theano.config.floatX)

        self.h_sample_given_x_func = \
            theano.function([self.x],
                    h_sample_given_x)

        if _PRINT_GRAPHS:
            print "----------------------\nh_sample_given_x_func"
            theano.printing.debugprint(self.h_sample_given_x_func)

        # propdown
        I_down = self._I_down(self.h)
        E_x_given_h = sigmoid(I_down)
        self.E_x_given_h_func = theano.function([self.h], E_x_given_h)

        if _PRINT_GRAPHS:
            print "----------------------\nE_x_given_h_func"
            theano.printing.debugprint(self.E_x_given_h_func)

        x_sample_given_h = \
            self.theano_rng.binomial( \
                            size = self._get_visibles_shape(),
                            n = 1, 
                            p = E_x_given_h, 
                            dtype = theano.config.floatX)

        self.x_sample_given_h_func = \
            theano.function([self.h], 
                    x_sample_given_h)

        if _PRINT_GRAPHS:
            print "----------------------\nx_sample_given_h_func"
            theano.printing.debugprint(self.x_sample_given_h_func)

        ##############################################
        # cd update done by grad of free energy
        
        x_tilde = T.tensor4('x_tilde') 
        cd_update_cost = self._mean_free_energy(self.x) - \
                            self._mean_free_energy(x_tilde)

        cd_grad = T.grad(cd_update_cost, self.params)
        # This is NLL minimization so we use a -
        cd_updates = {self.W: self.W - self.lr * cd_grad[0],
                    self.b_h: self.b_h - self.lr * cd_grad[1],
                    self.b_x: self.b_x - self.lr * cd_grad[2]}

        cd_returned = [cd_update_cost,
                        cd_grad[0], cd_grad[1], cd_grad[2],
                        self.lr * cd_grad[0],
                        self.lr * cd_grad[1],
                        self.lr * cd_grad[2]]
        self.cd_return_desc = \
            ['cd_update_cost',
                'cd_grad_W', 'cd_grad_b_h', 'cd_grad_b_x',
                'lr_times_cd_grad_W',
                'lr_times_cd_grad_b_h',
                'lr_times_cd_grad_b_x']
    
        self.cd_update_function = \
                theano.function([self.x, x_tilde], 
                    cd_returned, updates=cd_updates)

        if _PRINT_GRAPHS:
            print "----------------------\ncd_update_function"
            theano.printing.debugprint(self.cd_update_function)

        ##############
        # sparsity update, based on grad for b_h only

        '''
        This mean returns an array of shape 
            (num_hiddens_planes, feature_map_height, feature_map_width)
        (so it's a mean over each unit's activation)
        '''
        mean_expected_activation = T.mean(E_h_given_x, axis=0)
        # sparsity_p is broadcasted everywhere
        sparsity_update_cost = \
                T.sqr(self.sparsity_p - mean_expected_activation)
        sparsity_update_cost = \
            T.sum(T.sum(T.sum( \
                sparsity_update_cost, axis=2), axis=1), axis=0)
        sparsity_grad = T.grad(sparsity_update_cost, [self.W, self.b_h])

        sparsity_returned = \
            [sparsity_update_cost,
             sparsity_grad[0], sparsity_grad[1],
             self.sparsity_lambda * self.lr * sparsity_grad[0],
             self.sparsity_lambda * self.lr * sparsity_grad[1]]
        self.sparsity_return_desc = \
            ['sparsity_update_cost',
                'sparsity_grad_W',
                'sparsity_grad_b_h',
                'lambda_lr_times_sparsity_grad_W',
                'lambda_lr_times_sparsity_grad_b_h']

        # gradient _descent_ so we use a -
        sparsity_update = \
            {self.b_h: self.b_h - \
                self.sparsity_lambda * self.lr * sparsity_grad[1],
            self.W: self.W - \
                self.sparsity_lambda * self.lr * sparsity_grad[0]}
        self.sparsity_update_function = \
            theano.function([self.x], 
                sparsity_returned, updates=sparsity_update)

        if _PRINT_GRAPHS:
            print "----------------------\nsparsity_update_function"
            theano.printing.debugprint(self.sparsity_update_function)

    def CD_step(self, x):
        h1 = self.h_sample_given_x_func(x)
        x2 = self.x_sample_given_h_func(h1)
        return self.cd_update_function(x, x2)

    def sparsity_step(self, x):
        return self.sparsity_update_function(x)

    # these two also operate on minibatches

    def random_gibbs_samples(self, num_updown_steps):
        start_x = self.numpy_rng.rand(*self._get_visibles_shape())
        return self.gibbs_samples_from(start_x, num_updown_steps)

    def gibbs_samples_from(self, start_x, num_updown_steps):
        x_sample = start_x
        for i in xrange(num_updown_steps):
            h_sample = self.h_sample_given_x_func(x_sample)
            x_sample = self.x_sample_given_h_func(h_sample)
        return x_sample


