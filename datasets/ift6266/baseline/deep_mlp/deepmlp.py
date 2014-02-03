#

import numpy, cPickle, gzip


import theano
import theano.tensor as T

import time 

import theano.tensor.nnet

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model 
    that has one layer or more of hidden units and nonlinear activations. 
    Intermidiate layers usually have as activation function thanh or the 
    sigmoid function  while the top layer is a softamx layer. 
    """



    def __init__(self, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :param input: symbolic variable that describes the input of the 
        architecture (one minibatch)

        :param n_in: number of input units, the dimension of the space in 
        which the datapoints lie

        :param n_hidden: List representing the number of units for each 
		hidden layer
		
		#:param n_layer: Number of hidden layers

        :param n_out: number of output units, the dimension of the space in 
        which the labels lie

        """

        # initialize the parameters theta = (W,b) ; Here W and b are lists 
        # where W[i] and b[i] represent the parameters and the bias vector
        # of the i-th layer.
        n_layer=len(n_hidden)
        W_values=[]
        b_values=[]
        self.W=[]
        self.b=[]
		
	# We first initialize the matrix W[0] and b[0] that represent the parameters
	# from the input to the first hidden layer
        W_values.append(numpy.asarray( numpy.random.uniform( \
		      low = -numpy.sqrt(6./(n_in+n_hidden[0])), \
			  high = numpy.sqrt(6./(n_in+n_hidden[0])), \
			  size = (n_in, n_hidden[0])), dtype = theano.config.floatX))
        self.W.append(theano.shared( value = W_values[0] ))
        self.b.append(theano.shared( value = numpy.zeros((n_hidden[0],), 
                                                dtype= theano.config.floatX)))
												
        # We initialize the parameters between all consecutive hidden layers
        for i in range(1,n_layer):
        # Each `W[i]` is initialized with `W_values[i]` which is uniformely sampled
        # from -6./sqrt(n_hidden[i]+n_hidden[i+1]) and 6./sqrt(n_hidden[i]+n_hidden[i+1])
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
            W_values.append(numpy.asarray( numpy.random.uniform( \
		          low = -numpy.sqrt(6./(n_hidden[i-1]+n_hidden[i])), \
			      high = numpy.sqrt(6./(n_hidden[i-1]+n_hidden[i])), \
			      size = (n_hidden[i-1], n_hidden[i])), dtype = theano.config.floatX))
            self.W.append(theano.shared( value = W_values[i] ))
            self.b.append(theano.shared( value = numpy.zeros((n_hidden[i],), 
                                                dtype= theano.config.floatX)))

        # We initialize the matrix W[n_layer] and b[n_layer] that represent 
        # the parameters from the last hidden layer to the output layer using the
        # same uniform sampling.
        W_values.append(numpy.asarray( numpy.random.uniform( 
              low = -numpy.sqrt(6./(n_hidden[n_layer-1]+n_out)), \
              high= numpy.sqrt(6./(n_hidden[n_layer-1]+n_out)),\
              size= (n_hidden[n_layer-1], n_out)), dtype = theano.config.floatX))
        self.W.append(theano.shared( value = W_values[n_layer]))
        self.b.append(theano.shared( value = numpy.zeros((n_out,), 
                                                dtype= theano.config.floatX)))

        # List of the symbolic expressions computing the values each hidden layer
        self.hidden = []

	# Symbolic expression of the first hidden layer
        self.hidden.append(T.tanh(T.dot(input, self.W[0])+ self.b[0]))
        for i in range(1,n_layer):
	# Symbolic expression of the i-th hidden layer
            self.hidden.append(T.tanh(T.dot(self.hidden[i-1], self.W[i])+ self.b[i]))

        # symbolic expression computing the values of the top layer 
        self.p_y_given_x= T.nnet.softmax(T.dot(self.hidden[n_layer-1], self.W[n_layer])+self.b[n_layer])

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred = T.argmax( self.p_y_given_x, axis =1)
        
        # L1 norm ; one regularization option is to enforce L1 norm to 
        # be small 
        self.L1=abs(self.W[0]).sum()
        self.L2_sqr=abs(self.W[0]).sum()
        for i in range(1,n_layer+1):
            self.L1 += abs(self.W[i]).sum()
        # square of L2 norm ; one regularization option is to enforce 
        # square of L2 norm to be small
        for i in range(n_layer+1):
            self.L2_sqr += abs(self.W[i]**2).sum()

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch 
        over the total number of examples of the minibatch 
        """
 
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
def sgd_optimization_mnist( learning_rate=0.01, L1_reg = 0.00, \
                            L2_reg = 0.0001, n_iter=100,n_hidden=[200,100,90,80,70]):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer 
    perceptron

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used (factor for the stochastic 
    gradient

    :param L1_reg: L1-norm's weight when added to the cost (see 
    regularization)

    :param L2_reg: L2-norm's weight when added to the cost (see 
    regularization)
 
    :param n_iter: maximal number of iterations ot run the optimizer 

   """

    # Load the dataset 
    f = gzip.open('mnist.pkl.gz','rb')
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

    # construct the logistic regression class
    classifier = MLP( input=x.reshape((batch_size,28*28)),\
                      n_in=28*28, n_hidden=n_hidden, n_out=10)
    
    # the cost we minimize during training is the negative log likelihood of 
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr 

    # compiling a theano function that computes the mistakes that are made by 
    # the model on a minibatch
    test_model = theano.function([x,y], classifier.errors(y))
    g_W=[]
    g_b=[]
    # compute the gradient of cost with respect to theta = (W1, b1, W2, b2) 
    for i in range(len(n_hidden)+1):
        g_W.append(T.grad(cost, classifier.W[i]))
        g_b.append(T.grad(cost, classifier.b[i]))
	
	
    # specify how to update the parameters of the model as a dictionary
    updates={}
    for i in range(len(n_hidden)+1):
        updates[classifier.W[i]]= classifier.W[i] - learning_rate*g_W[i]
        updates[classifier.b[i]]= classifier.b[i] - learning_rate*g_b[i]
    # compiling a theano function `train_model` that returns the cost, but in 
    # the same time updates the parameter of the model based on the rules 
    # defined in `updates`
    train_model = theano.function([x, y], cost, updates = updates )
    n_minibatches        = len(train_batches) 
 
    # early-stopping parameters
    patience              = 10000 # look as this many examples regardless
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
        cost_ij = train_model(x,y)

        if (iter+1) % validation_frequency == 0: 
            # compute zero-one loss on validation set 
            this_validation_loss = 0.
            for x,y in valid_batches:
                # sum up the errors for each minibatch
                this_validation_loss += test_model(x,y)
            # get the average by dividing with the number of minibatches
            this_validation_loss /= len(valid_batches)

            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                   (epoch, minibatch_index+1, n_minibatches, \
                    this_validation_loss*100.))


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_score = 0.
                for x,y in test_batches:
                    test_score += test_model(x,y)
                test_score /= len(test_batches)
                print(('     epoch %i, minibatch %i/%i, test error of best '
                      'model %f %%') % 
                             (epoch, minibatch_index+1, n_minibatches,
                              test_score*100.))

        if patience <= iter :
            break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %  
                 (best_validation_loss * 100., best_iter, test_score*100.))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))		
    #test on NIST (you need pylearn and access to NIST to do that)
if __name__ == '__main__':
    sgd_optimization_mnist()
							   
