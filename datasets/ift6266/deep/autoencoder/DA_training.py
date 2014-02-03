"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SDAE. 
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting 
 latent representation y is then mapped back to a "reconstructed" vector 
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight 
 matrix W' can optionally be constrained such that W' = W^T, in which case 
 the autoencoder is said to have tied weights. The network is trained such 
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into 
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means 
 of a stochastic mapping. Afterwards y is computed as before (using 
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction 
 error is now measured between z and the uncorrupted input x, which is 
 computed as the cross-entropy : 
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]

 For X iteration of the main program loop it takes *** minutes on an 
 Intel Core i7 and *** minutes on GPU (NVIDIA GTX 285 graphics processor).


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and 
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing 
   Systems 19, 2007

"""

import numpy 
import theano
import time
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import gzip
import cPickle

from pylearn.io import filetensor as ft

class dA():
  """Denoising Auto-Encoder class (dA) 

  A denoising autoencoders tries to reconstruct the input from a corrupted 
  version of it by projecting it first in a latent space and reprojecting 
  it afterwards back in the input space. Please refer to Vincent et al.,2008
  for more details. If x is the input then equation (1) computes a partially
  destroyed version of x by means of a stochastic mapping q_D. Equation (2) 
  computes the projection of the input into the latent space. Equation (3) 
  computes the reconstruction of the input, while equation (4) computes the 
  reconstruction error.
  
  .. math::

    \tilde{x} ~ q_D(\tilde{x}|x)                                         (1)

    y = s(W \tilde{x} + b)                                               (2)

    z = s(W' y  + b')                                                    (3)

    L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]          (4)

  """

  def __init__(self, n_visible= 784, n_hidden= 500, complexity = 0.1, input= None):
        """
        Initialize the DAE class by specifying the number of visible units (the 
        dimension d of the input ), the number of hidden units ( the dimension 
        d' of the latent or hidden space ) and by giving a symbolic variable 
        for the input. Such a symbolic variable is useful when the input is 
        the result of some computations. For example when dealing with SDAEs,
        the dA on layer 2 gets as input the output of the DAE on layer 1. 
        This output can be written as a function of the input to the entire 
        model, and as such can be computed by theano whenever needed. 
        
        :param n_visible: number of visible units

        :param n_hidden:  number of hidden units

        :param input:     a symbolic description of the input or None 

        """
        self.n_visible = n_visible
        self.n_hidden  = n_hidden
        
        # create a Theano random generator that gives symbolic random values
        theano_rng = RandomStreams()
        # create a numpy random generator
        numpy_rng = numpy.random.RandomState()
		
        # print the parameter of the DA
        if True :
            print 'input size = %d' %n_visible
            print 'hidden size = %d' %n_hidden
            print 'complexity = %2.2f' %complexity
         
        # initial values for weights and biases
        # note : W' was written as `W_prime` and b' as `b_prime`

        # W is initialized with `initial_W` which is uniformely sampled
        # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        initial_W = numpy.asarray( numpy.random.uniform( \
                  low = -numpy.sqrt(6./(n_visible+n_hidden)), \
                  high = numpy.sqrt(6./(n_visible+n_hidden)), \
                  size = (n_visible, n_hidden)), dtype = theano.config.floatX)
        initial_b       = numpy.zeros(n_hidden)

        # W' is initialized with `initial_W_prime` which is uniformely sampled
        # from -6./sqrt(n_visible+n_hidden) and 6./sqrt(n_hidden+n_visible)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        initial_b_prime= numpy.zeros(n_visible)
         
        
        # theano shared variables for weights and biases
        self.W       = theano.shared(value = initial_W,       name = "W")
        self.b       = theano.shared(value = initial_b,       name = "b")
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T 
        self.b_prime = theano.shared(value = initial_b_prime, name = "b'")

        # if no input is given, generate a variable representing the input
        if input == None : 
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            x = T.dmatrix(name = 'input') 
        else:
            x = input
        # Equation (1)
        # note : first argument of theano.rng.binomial is the shape(size) of 
        #        random numbers that it should produce
        #        second argument is the number of trials 
        #        third argument is the probability of success of any trial
        #
        #        this will produce an array of 0s and 1s where 1 has a 
        #        probability of 0.9 and 0 of 0.1

        tilde_x  = theano_rng.binomial( x.shape,  1,  1-complexity) * x
        # Equation (2)
        # note  : y is stored as an attribute of the class so that it can be 
        #         used later when stacking dAs. 
        self.y   = T.nnet.sigmoid(T.dot(tilde_x, self.W      ) + self.b)
        # Equation (3)
        z        = T.nnet.sigmoid(T.dot(self.y, self.W_prime) + self.b_prime)
        # Equation (4)
        self.L = - T.sum( x*T.log(z) + (1-x)*T.log(1-z), axis=1 ) 
        # note : L is now a vector, where each element is the cross-entropy cost 
        #        of the reconstruction of the corresponding example of the 
        #        minibatch. We need to compute the average of all these to get 
        #        the cost of the minibatch
        self.cost = T.mean(self.L)
        # note : y is computed from the corrupted `tilde_x`. Later on, 
        #        we will need the hidden layer obtained from the uncorrupted 
        #        input when for example we will pass this as input to the layer 
        #        above
        self.hidden_values = T.nnet.sigmoid( T.dot(x, self.W) + self.b)



def sgd_optimization_nist( learning_rate=0.01,  \
                            n_iter = 300, n_code_layer = 400, \
                            complexity = 0.1):
    """
    Demonstrate stochastic gradient descent optimization for a denoising autoencoder

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used (factor for the stochastic 
    gradient

    :param pretraining_epochs: number of epoch to do pretraining

    :param pretrain_lr: learning rate to be used during pre-training

    :param n_iter: maximal number of iterations ot run the optimizer 

    """
    #open file to save the validation and test curve
    filename = 'lr_' + str(learning_rate) + 'ni_' + str(n_iter) + 'nc_' + str(n_code_layer) + \
    'c_' + str(complexity) + '.txt'

    result_file = open(filename, 'w')



    data_path = '/data/lisa/data/nist/by_class/'
    f = open(data_path+'all/all_train_data.ft')
    g = open(data_path+'all/all_train_labels.ft')
    h = open(data_path+'all/all_test_data.ft')
    i = open(data_path+'all/all_test_labels.ft')
    
    train_set_x = ft.read(f)
    train_set_y = ft.read(g)
    test_set_x = ft.read(h)
    test_set_y = ft.read(i)
    
    f.close()
    g.close()
    i.close()
    h.close()

    # make minibatches of size 20 
    batch_size = 20    # sized of the minibatch

    #create a validation set the same size as the test size
    #use the end of the training array for this purpose
    #discard the last remaining so we get a %batch_size number
    test_size=len(test_set_y)
    test_size = int(test_size/batch_size)
    test_size*=batch_size
    train_size = len(train_set_x)
    train_size = int(train_size/batch_size)
    train_size*=batch_size
    validation_size =test_size 
    offset = train_size-test_size
    if True:
        print 'train size = %d' %train_size
        print 'test size = %d' %test_size
        print 'valid size = %d' %validation_size
        print 'offset = %d' %offset
    
    
    #train_set = (train_set_x,train_set_y)
    train_batches = []
    for i in xrange(0, train_size-test_size, batch_size):
        train_batches = train_batches + \
            [(train_set_x[i:i+batch_size], train_set_y[i:i+batch_size])]
            
    test_batches = []
    for i in xrange(0, test_size, batch_size):
        test_batches = test_batches + \
            [(test_set_x[i:i+batch_size], test_set_y[i:i+batch_size])]
    
    valid_batches = []
    for i in xrange(0, test_size, batch_size):
        valid_batches = valid_batches + \
            [(train_set_x[offset+i:offset+i+batch_size], \
            train_set_y[offset+i:offset+i+batch_size])]


    ishape     = (32,32) # this is the size of NIST images

    # allocate symbolic variables for the data
    x = T.fmatrix()  # the data is presented as rasterized images
    y = T.lvector()  # the labels are presented as 1D vector of 
                          # [long int] labels

    # construct the denoising autoencoder class
    n_ins = 32*32
    encoder = dA(n_ins, n_code_layer, complexity, input = x.reshape((batch_size,n_ins)))

    # Train autoencoder
    
    # compute gradients of the layer parameters
    gW       = T.grad(encoder.cost, encoder.W)
    gb       = T.grad(encoder.cost, encoder.b)
    gb_prime = T.grad(encoder.cost, encoder.b_prime)
    # compute the updated value of the parameters after one step
    updated_W       = encoder.W       - gW       * learning_rate
    updated_b       = encoder.b       - gb       * learning_rate
    updated_b_prime = encoder.b_prime - gb_prime * learning_rate

    # defining the function that evaluate the symbolic description of 
    # one update step 
    train_model = theano.function([x], encoder.cost, updates=\
                    { encoder.W       : updated_W, \
                      encoder.b       : updated_b, \
                      encoder.b_prime : updated_b_prime } )


 

    # compiling a theano function that computes the mistakes that are made  
    # by the model on a minibatch
    test_model = theano.function([x], encoder.cost)

    normalize = numpy.asarray(255, dtype=theano.config.floatX)

  
    n_minibatches        = len(train_batches) 
 
    # early-stopping parameters
    patience              = 10000000 / batch_size # look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = n_minibatches  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 


    best_params          = None
    best_validation_loss = float('inf')
    best_iter            = 0
    test_score           = 0.
    start_time = time.clock()
    # have a maximum of `n_iter` iterations through the entire dataset
    for iter in xrange(n_iter* n_minibatches):

        # get epoch and minibatch index
        epoch           = iter / n_minibatches
        minibatch_index =  iter % n_minibatches

        # get the minibatches corresponding to `iter` modulo
        # `len(train_batches)`
        x,y = train_batches[ minibatch_index ]
        '''
        if iter == 0:
            b = numpy.asarray(255, dtype=theano.config.floatX)
            x = x / b
            print x
            print y
            print x.__class__
            print x.shape
            print x.dtype.name
            print y.dtype.name
            print x.min(), x.max()
        '''
        
        cost_ij = train_model(x/normalize)

        if (iter+1) % validation_frequency == 0: 
            # compute zero-one loss on validation set 
            this_validation_loss = 0.
            for x,y in valid_batches:
                # sum up the errors for each minibatch
                this_validation_loss += test_model(x/normalize)
            # get the average by dividing with the number of minibatches
            this_validation_loss /= len(valid_batches)

            print('epoch %i, minibatch %i/%i, validation error %f ' % \
                   (epoch, minibatch_index+1, n_minibatches, \
                    this_validation_loss))

            # save value in file
            result_file.write(str(epoch) + ' ' + str(this_validation_loss)+ '\n')


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter
                # test it on the test set
            
                test_score = 0.
                for x,y in test_batches:
                    test_score += test_model(x/normalize)
                test_score /= len(test_batches)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f ') % 
                             (epoch, minibatch_index+1, n_minibatches,
                              test_score))

        if patience <= iter :
                print('iter (%i) is superior than patience(%i). break', (iter, patience))
                break

        

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f ,'
           'with test performance %f ') %  
                 (best_validation_loss, test_score))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))

    
    result_file.close()

    return (best_validation_loss, test_score, (end_time-start_time)/60, best_iter)

def sgd_optimization_mnist( learning_rate=0.01,  \
                            n_iter = 1, n_code_layer = 400, \
                            complexity = 0.1):
    """
    Demonstrate stochastic gradient descent optimization for a denoising autoencoder

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used (factor for the stochastic 
    gradient

    :param pretraining_epochs: number of epoch to do pretraining

    :param pretrain_lr: learning rate to be used during pre-training

    :param n_iter: maximal number of iterations ot run the optimizer 

    """
    #open file to save the validation and test curve
    filename = 'lr_' + str(learning_rate) + 'ni_' + str(n_iter) + 'nc_' + str(n_code_layer) + \
    'c_' + str(complexity) + '.txt'

    result_file = open(filename, 'w')

    # Load the dataset 
    f = gzip.open('/u/lisa/HTML/deep/data/mnist/mnist.pkl.gz','rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # make minibatches of size 20 
    batch_size = 20    # sized of the minibatch

    # Dealing with the training set
    # get the list of training images (x) and their labels (y)
    (train_set_x, train_set_y) = train_set
    # initialize the list of training minibatches with empty list
    train_batches = []
    for i in xrange(0, len(train_set_x), batch_size):
        # add to the list of minibatches the minibatch starting at 
        # position i, ending at position i+batch_size
        # a minibatch is a pair ; the first element of the pair is a list 
        # of datapoints, the second element is the list of corresponding 
        # labels
        train_batches = train_batches + \
               [(train_set_x[i:i+batch_size], train_set_y[i:i+batch_size])]

    # Dealing with the validation set
    (valid_set_x, valid_set_y) = valid_set
    # initialize the list of validation minibatches 
    valid_batches = []
    for i in xrange(0, len(valid_set_x), batch_size):
        valid_batches = valid_batches + \
               [(valid_set_x[i:i+batch_size], valid_set_y[i:i+batch_size])]

    # Dealing with the testing set
    (test_set_x, test_set_y) = test_set
    # initialize the list of testing minibatches 
    test_batches = []
    for i in xrange(0, len(test_set_x), batch_size):
        test_batches = test_batches + \
              [(test_set_x[i:i+batch_size], test_set_y[i:i+batch_size])]


    ishape     = (28,28) # this is the size of MNIST images

    # allocate symbolic variables for the data
    x = T.fmatrix()  # the data is presented as rasterized images
    y = T.lvector()  # the labels are presented as 1D vector of 
                          # [long int] labels

    # construct the denoising autoencoder class
    n_ins = 28*28
    encoder = dA(n_ins, n_code_layer, complexity, input = x.reshape((batch_size,n_ins)))

    # Train autoencoder
    
    # compute gradients of the layer parameters
    gW       = T.grad(encoder.cost, encoder.W)
    gb       = T.grad(encoder.cost, encoder.b)
    gb_prime = T.grad(encoder.cost, encoder.b_prime)
    # compute the updated value of the parameters after one step
    updated_W       = encoder.W       - gW       * learning_rate
    updated_b       = encoder.b       - gb       * learning_rate
    updated_b_prime = encoder.b_prime - gb_prime * learning_rate

    # defining the function that evaluate the symbolic description of 
    # one update step 
    train_model = theano.function([x], encoder.cost, updates=\
                    { encoder.W       : updated_W, \
                      encoder.b       : updated_b, \
                      encoder.b_prime : updated_b_prime } )


 

    # compiling a theano function that computes the mistakes that are made  
    # by the model on a minibatch
    test_model = theano.function([x], encoder.cost)



  
    n_minibatches        = len(train_batches) 
 
    # early-stopping parameters
    patience              = 10000# look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = n_minibatches  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 


    best_params          = None
    best_validation_loss = float('inf')
    best_iter            = 0
    test_score           = 0.
    start_time = time.clock()
    # have a maximum of `n_iter` iterations through the entire dataset
    for iter in xrange(n_iter* n_minibatches):

        # get epoch and minibatch index
        epoch           = iter / n_minibatches
        minibatch_index =  iter % n_minibatches

        # get the minibatches corresponding to `iter` modulo
        # `len(train_batches)`
        x,y = train_batches[ minibatch_index ]
        cost_ij = train_model(x)

        if (iter+1) % validation_frequency == 0: 
            # compute zero-one loss on validation set 
            this_validation_loss = 0.
            for x,y in valid_batches:
                # sum up the errors for each minibatch
                this_validation_loss += test_model(x)
            # get the average by dividing with the number of minibatches
            this_validation_loss /= len(valid_batches)

            print('epoch %i, minibatch %i/%i, validation error %f ' % \
                   (epoch, minibatch_index+1, n_minibatches, \
                    this_validation_loss))

            # save value in file
            result_file.write(str(epoch) + ' ' + str(this_validation_loss)+ '\n')


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter
                # test it on the test set
            
                test_score = 0.
                for x,y in test_batches:
                    test_score += test_model(x)
                test_score /= len(test_batches)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f ') % 
                             (epoch, minibatch_index+1, n_minibatches,
                              test_score))

        if patience <= iter :
                print('iter (%i) is superior than patience(%i). break', iter, patience)
                break


    end_time = time.clock()
    print(('Optimization complete with best validation score of %f ,'
           'with test performance %f ') %  
                 (best_validation_loss, test_score))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))

    
    result_file.close()

    return (best_validation_loss, test_score, (end_time-start_time)/60, best_iter)


def experiment(state,channel):

    (best_validation_loss, test_score, minutes_trained, iter) = \
        sgd_optimization_mnist(state.learning_rate, state.n_iter, state.n_code_layer,
            state.complexity)

    state.best_validation_loss = best_validation_loss
    state.test_score = test_score
    state.minutes_trained = minutes_trained
    state.iter = iter

    return channel.COMPLETE

def experiment_nist(state,channel):

    (best_validation_loss, test_score, minutes_trained, iter) = \
        sgd_optimization_nist(state.learning_rate, state.n_iter, state.n_code_layer,
            state.complexity)

    state.best_validation_loss = best_validation_loss
    state.test_score = test_score
    state.minutes_trained = minutes_trained
    state.iter = iter

    return channel.COMPLETE


if __name__ == '__main__':

    sgd_optimization_nist()


