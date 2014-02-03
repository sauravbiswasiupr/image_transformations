import numpy
import theano
import time
import sys
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.softsign
import copy
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv 

sys.path.append('../../')
#import ift6266.datasets
import ift6266.datasets
from ift6266.baseline.log_reg.log_reg import LogisticRegression

from theano.tensor.xlogx import xlogx, xlogy0
# it's target*log(output)
def binary_cross_entropy(target, output, sum_axis=1):
    XE = xlogy0(target, output) + xlogy0((1 - target), (1 - output))
    return -T.sum(XE, axis=sum_axis)



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
 
        self.output = T.tanh(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]
 
class dA_conv(object):
 
  def __init__(self, input, filter_shape, corruption_level = 0.1, 
               shared_W = None, shared_b = None, image_shape = None, num = 0,batch_size=20):

    theano_rng = RandomStreams()
    
    fan_in = numpy.prod(filter_shape[1:])
    fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])

    center = theano.shared(value = 1, name="center")
    scale = theano.shared(value = 2, name="scale")

    if shared_W != None and shared_b != None :
        self.W = shared_W
        self.b = shared_b
    else:
        initial_W = numpy.asarray( numpy.random.uniform(
              low = -numpy.sqrt(6./(fan_in+fan_out)),
              high = numpy.sqrt(6./(fan_in+fan_out)),
              size = filter_shape), dtype = theano.config.floatX)
        initial_b = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.W = theano.shared(value = initial_W, name = "W")
        self.b = theano.shared(value = initial_b, name = "b")
    
 
    initial_b_prime= numpy.zeros((filter_shape[1],),dtype=theano.config.floatX)

    self.b_prime = theano.shared(value = initial_b_prime, name = "b_prime")
 
    self.x = input

    self.tilde_x = theano_rng.binomial( self.x.shape, 1, 1 - corruption_level,dtype=theano.config.floatX) * self.x

    conv1_out = conv.conv2d(self.tilde_x, self.W, filter_shape=filter_shape,
                            image_shape=image_shape,
                            unroll_kern=4,unroll_batch=4, 
                            border_mode='valid')

    
    self.y = T.tanh(conv1_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    
    da_filter_shape = [ filter_shape[1], filter_shape[0], filter_shape[2],\
                       filter_shape[3] ]
    da_image_shape = [ batch_size, filter_shape[0], image_shape[2]-filter_shape[2]+1, 
                       image_shape[3]-filter_shape[3]+1 ]
    #import pdb; pdb.set_trace()
    initial_W_prime =  numpy.asarray( numpy.random.uniform( \
              low = -numpy.sqrt(6./(fan_in+fan_out)), \
              high = numpy.sqrt(6./(fan_in+fan_out)), \
              size = da_filter_shape), dtype = theano.config.floatX)
    self.W_prime = theano.shared(value = initial_W_prime, name = "W_prime")

    conv2_out = conv.conv2d(self.y, self.W_prime,
                            filter_shape = da_filter_shape,\
                            image_shape = da_image_shape, \
                            unroll_kern=4,unroll_batch=4, \
                            border_mode='full')

    self.z =  (T.tanh(conv2_out + self.b_prime.dimshuffle('x', 0, 'x', 'x'))+center) / scale
    
    if num != 0 :
        scaled_x = (self.x + center) / scale
    else: 
        scaled_x = self.x
    self.L = - T.sum( scaled_x*T.log(self.z) + (1-scaled_x)*T.log(1-self.z), axis=1 )

    self.cost = T.mean(self.L)

    self.params = [ self.W, self.b, self.b_prime ] 

class LeNetConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape=None, poolsize=(2,2)):
        self.input = input
  
        W_values = numpy.zeros(filter_shape, dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values)
 
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values)
 
        conv_out = conv.conv2d(input, self.W,
                filter_shape=filter_shape, image_shape=image_shape,
                               unroll_kern=4,unroll_batch=4)
 

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize)

        W_bound = numpy.sqrt(6./(fan_in + fan_out))
        self.W.value = numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype = theano.config.floatX)
  

        pooled_out = downsample.max_pool2D(conv_out, poolsize, ignore_border=True)
 
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
 

class CSdA():
    def __init__(self, n_ins_mlp,batch_size, conv_hidden_layers_sizes,
                 mlp_hidden_layers_sizes, corruption_levels, rng, n_out, 
                 pretrain_lr, finetune_lr):

        # Just to make sure those are not modified somewhere else afterwards
        hidden_layers_sizes = copy.deepcopy(mlp_hidden_layers_sizes)
        corruption_levels = copy.deepcopy(corruption_levels)

        #update_locals(self, locals())


        
        self.layers = []
        self.pretrain_functions = []
        self.params = []
        self.n_layers = len(conv_hidden_layers_sizes)
        self.mlp_n_layers = len(mlp_hidden_layers_sizes)
        
        self.x = T.matrix('x') # the data is presented as rasterized images
        self.y = T.ivector('y') # the labels are presented as 1D vector of
        
        for i in xrange( self.n_layers ):
            filter_shape=conv_hidden_layers_sizes[i][0]
            image_shape=conv_hidden_layers_sizes[i][1]
            max_poolsize=conv_hidden_layers_sizes[i][2]
                
            if i == 0 :
                layer_input=self.x.reshape((batch_size, 1, 32, 32))
            else:
                layer_input=self.layers[-1].output
            
            layer = LeNetConvPoolLayer(rng, input=layer_input,
                                       image_shape=image_shape,
                                       filter_shape=filter_shape,
                                       poolsize=max_poolsize)
            print 'Convolutional layer', str(i+1), 'created'
            
            self.layers += [layer]
            self.params += layer.params

            da_layer = dA_conv(corruption_level = corruption_levels[0],
                               input = layer_input,
                               shared_W = layer.W, shared_b = layer.b,
                               filter_shape=filter_shape,
                               image_shape = image_shape, num=i , batch_size=batch_size)
            
            gparams = T.grad(da_layer.cost, da_layer.params)
            
            updates = {}
            for param, gparam in zip(da_layer.params, gparams):
                updates[param] = param - gparam * pretrain_lr
            
            update_fn = theano.function([self.x], da_layer.cost, updates = updates)
            
            self.pretrain_functions += [update_fn]
            
        for i in xrange( self.mlp_n_layers ): 
            if i == 0 :
                input_size = n_ins_mlp
            else:
                input_size = mlp_hidden_layers_sizes[i-1]
            
            if i == 0 :
                if len( self.layers ) == 0 :
                    layer_input=self.x
                else :
                    layer_input = self.layers[-1].output.flatten(2)
            else:
                layer_input = self.layers[-1].output
            
            layer = SigmoidalLayer(rng, layer_input, input_size,
                                        mlp_hidden_layers_sizes[i] )
            
            self.layers += [layer]
            self.params += layer.params
            
            print 'MLP layer', str(i+1), 'created'
            
        self.logLayer = LogisticRegression(input=self.layers[-1].output, \
                                                     n_in=mlp_hidden_layers_sizes[-1], n_out=n_out)


        self.params += self.logLayer.params
        self.all_params = self.params
        cost = self.logLayer.negative_log_likelihood(self.y)
        
        gparams = T.grad(cost, self.params)

        updates = {}
        for param,gparam in zip(self.params, gparams):
            updates[param] = param - gparam*finetune_lr
        
        self.finetune = theano.function([self.x, self.y], cost, updates = updates)
        
        self.errors = self.logLayer.errors(self.y)



