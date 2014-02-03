import numpy
import theano
import time
import sys
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
#import theano.sandbox.softsign

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv 

from ift6266 import datasets
from ift6266.baseline.log_reg.log_reg import LogisticRegression

batch_size = 100

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
               shared_W = None, shared_b = None, image_shape = None, 
               poolsize = (2,2)):

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
                            image_shape=image_shape, border_mode='valid')
    
    self.y = T.tanh(conv1_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    
    da_filter_shape = [ filter_shape[1], filter_shape[0], 
                        filter_shape[2], filter_shape[3] ]
    initial_W_prime =  numpy.asarray( numpy.random.uniform( \
              low = -numpy.sqrt(6./(fan_in+fan_out)), \
              high = numpy.sqrt(6./(fan_in+fan_out)), \
              size = da_filter_shape), dtype = theano.config.floatX)
    self.W_prime = theano.shared(value = initial_W_prime, name = "W_prime")

    conv2_out = conv.conv2d(self.y, self.W_prime,
                            filter_shape = da_filter_shape,
                            border_mode='full')

    self.z =  (T.tanh(conv2_out + self.b_prime.dimshuffle('x', 0, 'x', 'x'))+center) / scale

    scaled_x = (self.x + center) / scale

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
                filter_shape=filter_shape, image_shape=image_shape)
 

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize)

        W_bound = numpy.sqrt(6./(fan_in + fan_out))
        self.W.value = numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype = theano.config.floatX)
  

        pooled_out = downsample.max_pool2D(conv_out, poolsize, ignore_border=True)
 
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
 

class SdA():
    def __init__(self, input, n_ins_mlp, conv_hidden_layers_sizes,
                 mlp_hidden_layers_sizes, corruption_levels, rng, n_out, 
                 pretrain_lr, finetune_lr, img_shape):
        
        self.layers = []
        self.pretrain_functions = []
        self.params = []
        self.conv_n_layers = len(conv_hidden_layers_sizes)
        self.mlp_n_layers = len(mlp_hidden_layers_sizes)
        
        self.x = T.matrix('x') # the data is presented as rasterized images
        self.y = T.ivector('y') # the labels are presented as 1D vector of
        
        for i in xrange( self.conv_n_layers ):
            filter_shape=conv_hidden_layers_sizes[i][0]
            image_shape=conv_hidden_layers_sizes[i][1]
            max_poolsize=conv_hidden_layers_sizes[i][2]
                
            if i == 0 :
                layer_input=self.x.reshape((self.x.shape[0], 1) + img_shape)
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
                               filter_shape = filter_shape,
                               image_shape = image_shape )
            
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
        
        cost = self.logLayer.negative_log_likelihood(self.y)
        
        gparams = T.grad(cost, self.params)

        updates = {}
        for param,gparam in zip(self.params, gparams):
            updates[param] = param - gparam*finetune_lr
        
        self.finetune = theano.function([self.x, self.y], cost, updates = updates)
        
        self.errors = self.logLayer.errors(self.y)

def sgd_optimization_mnist(learning_rate=0.1, pretraining_epochs = 1,
                           pretrain_lr = 0.1, training_epochs = 1000,
                           kernels = [[4,5,5], [4,3,3]], mlp_layers=[500],
                           corruption_levels = [0.2, 0.2, 0.2], 
                           batch_size = batch_size, img_shape=(28, 28),
                           max_pool_layers = [[2,2], [2,2]],
                           dataset=datasets.mnist(5000)):
 
    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x') # the data is presented as rasterized images
    y = T.ivector('y') # the labels are presented as 1d vector of
    # [int] labels

    layer0_input = x.reshape((x.shape[0],1)+img_shape)
    
    rng = numpy.random.RandomState(1234)
    conv_layers=[]
    init_layer = [[kernels[0][0],1,kernels[0][1],kernels[0][2]],
                  None, # do not specify the batch size since it can 
                        # change for the last one and then theano will 
                        # crash.
                  max_pool_layers[0]]
    conv_layers.append(init_layer)

    conv_n_out = (img_shape[0]-kernels[0][2]+1)/max_pool_layers[0][0]

    for i in range(1,len(kernels)):    
        layer = [[kernels[i][0],kernels[i-1][0],kernels[i][1],kernels[i][2]],
                 None, # same comment as for init_layer
                 max_pool_layers[i] ]
        conv_layers.append(layer)
        conv_n_out =  (conv_n_out - kernels[i][2]+1)/max_pool_layers[i][0]

    network = SdA(input = layer0_input, n_ins_mlp = kernels[-1][0]*conv_n_out**2,
                  conv_hidden_layers_sizes = conv_layers,
                  mlp_hidden_layers_sizes = mlp_layers,
                  corruption_levels = corruption_levels, n_out = 62,
                  rng = rng , pretrain_lr = pretrain_lr,
                  finetune_lr = learning_rate, img_shape=img_shape)

    test_model = theano.function([network.x, network.y], network.errors)
 
    start_time = time.clock()
    for i in xrange(len(network.layers)-len(mlp_layers)):
        for epoch in xrange(pretraining_epochs):
            for x, y in dataset.train(batch_size):
                c = network.pretrain_functions[i](x)
            print 'pre-training convolution layer %i, epoch %d, cost '%(i,epoch), c

    patience = 10000 # look as this many examples regardless
    patience_increase = 2. # WAIT THIS MUCH LONGER WHEN A NEW BEST IS
                                  # FOUND
    improvement_threshold = 0.995 # a relative improvement of this much is

    validation_frequency = patience/2
 
    best_params = None
    best_validation_loss = float('inf')
    test_score = 0.
    start_time = time.clock()
 
    done_looping = False
    epoch = 0
    iter = 0

    while (epoch < training_epochs) and (not done_looping):
      epoch = epoch + 1
      for x, y in dataset.train(batch_size):
 
        cost_ij = network.finetune(x, y)
        iter += 1
        
        if iter % validation_frequency == 0:
            validation_losses = [test_model(xv, yv) for xv, yv in dataset.valid(batch_size)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, iter %i, validation error %f %%' % \
                   (epoch, iter, this_validation_loss*100.))
            
            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
 
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)
                
                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter
                
                # test it on the test set
                test_losses = [test_model(xt, yt) for xt, yt in dataset.test(batch_size)]
                test_score = numpy.mean(test_losses)
                print((' epoch %i, iter %i, test error of best '
                      'model %f %%') %
                             (epoch, iter, test_score*100.))
                
        if patience <= iter :
            done_looping = True
            break
    
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score*100.))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))
 
if __name__ == '__main__':
    sgd_optimization_mnist()
 
