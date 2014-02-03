"""
This tutorial introduces logistic regression using Theano and stochastic 
gradient descent.  

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability. 

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of 
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method 
suitable for large datasets, and a conjugate gradient optimization method 
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" - 
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import numpy, time

import theano
import theano.tensor as T
from ift6266 import datasets

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W` 
    and bias vector :math:`b`. Classification is done by projecting data 
    points onto a set of hyperplanes, the distance to which is used to 
    determine a class membership probability. 
    """


    def __init__( self, input, n_in, n_out ):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the 
                      architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in 
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in 
                      which the labels lie

        """ 

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out) 
        self.W = theano.shared( value = numpy.zeros(( n_in, n_out ), dtype = theano.config.floatX ),
                                name =' W')
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared( value = numpy.zeros(( n_out, ), dtype = theano.config.floatX ),
                               name = 'b')


        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax( T.dot( input, self.W ) + self.b )

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred=T.argmax( self.p_y_given_x, axis =1 )

        # parameters of the model
        self.params = [ self.W, self.b ]


    def negative_log_likelihood( self, y ):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and one column per class 
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        # and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean( T.log( self.p_y_given_x )[ T.arange( y.shape[0] ), y ] )

    def MSE(self, y):
        return -T.mean(abs((self.p_t_given_x)[T.arange(y.shape[0]), y]-y)**2)

    def errors( self, y ):
        """Return a float representing the number of errors in the minibatch 
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the 
                  correct label
        """

        # check if y has same dimension of y_pred 
        if y.ndim != self.y_pred.ndim:
            raise TypeError( 'y should have the same shape as self.y_pred', 
                ( 'y', target.type, 'y_pred', self.y_pred.type ) )
        # check if y is of the correct datatype        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean( T.neq( self.y_pred, y ) )
        else:
            raise NotImplementedError()
        
#--------------------------------------------------------------------------------------------------------------------
# MAIN
#--------------------------------------------------------------------------------------------------------------------

def log_reg( learning_rate = 0.13, nb_max_examples =1000000, batch_size = 50, \
                    dataset=datasets.nist_digits(), image_size = 32 * 32, nb_class = 10,  \
                    patience = 5000, patience_increase = 2, improvement_threshold = 0.995):
    
    #28 * 28 = 784
    """
    Demonstrate stochastic gradient descent optimization of a log-linear 
    model

    This is demonstrated on MNIST.
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic 
                          gradient)

    :type nb_max_examples: int
    :param nb_max_examples: maximal number of epochs to run the optimizer 
    
    :type batch_size: int  
    :param batch_size:  size of the minibatch

    :type dataset: dataset
    :param dataset: a dataset instance from ift6266.datasets
                        
    :type image_size: int
    :param image_size: size of the input image in pixels (width * height)
    
    :type nb_class: int
    :param nb_class: number of classes
    
    :type patience: int
    :param patience: look as this many examples regardless
    
    :type patience_increase: int
    :param patience_increase: wait this much longer when a new best is found
    
    :type improvement_threshold: float
    :param improvement_threshold: a relative improvement of this much is considered significant


    """
    #--------------------------------------------------------------------------------------------------------------------
    # Build actual model
    #--------------------------------------------------------------------------------------------------------------------
    
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar( )    # index to a [mini]batch 
    x        = T.matrix('x')  # the data is presented as rasterized images
    y        = T.ivector('y') # the labels are presented as 1D vector of 
                           # [int] labels

    # construct the logistic regression class
    
    classifier = LogisticRegression( input = x, n_in = image_size, n_out = nb_class )

    # the cost we minimize during training is the negative log likelihood of 
    # the model in symbolic format
    cost = classifier.negative_log_likelihood( y ) 

    # compiling a Theano function that computes the mistakes that are made by 
    # the model on a minibatch
    test_model = theano.function( inputs = [ x, y ], 
            outputs = classifier.errors( y ))

    validate_model = theano.function( inputs = [ x, y ], 
            outputs = classifier.errors( y ))

    # compute the gradient of cost with respect to theta = ( W, b ) 
    g_W = T.grad( cost = cost, wrt = classifier.W )
    g_b  = T.grad( cost = cost, wrt = classifier.b )

    # specify how to update the parameters of the model as a dictionary
    updates = { classifier.W: classifier.W - learning_rate * g_W,\
                         classifier.b: classifier.b  - learning_rate * g_b}

    # compiling a Theano function `train_model` that returns the cost, but in 
    # the same time updates the parameter of the model based on the rules 
    # defined in `updates`
    train_model = theano.function( inputs = [ x, y ], 
            outputs = cost, 
            updates = updates)

    #--------------------------------------------------------------------------------------------------------------------
    # Train model
    #--------------------------------------------------------------------------------------------------------------------
   
    print '... training the model'
    # early-stopping parameters
    patience              = 5000  # look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = patience * 0.5
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 

    best_params          = None
    best_validation_loss = float('inf')
    test_score           = 0.
    start_time           = time.clock()

    done_looping = False 
    n_iters      = nb_max_examples / batch_size
    epoch        = 0
    iter        = 0
    
    while ( iter < n_iters ) and ( not done_looping ):
        
      epoch = epoch + 1
      for x, y in dataset.train(batch_size):

        minibatch_avg_cost = train_model( x, y )
        # iteration number
        iter += 1

        if iter % validation_frequency == 0: 
            # compute zero-one loss on validation set 
            validation_losses     = [ validate_model( xv, yv ) for xv, yv in dataset.valid(batch_size) ]
            this_validation_loss = numpy.mean( validation_losses )

            print('epoch %i, iter %i, validation error %f %%' % \
                 ( epoch, iter, this_validation_loss*100. ) )


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max( patience, iter * patience_increase )

                best_validation_loss = this_validation_loss
                # test it on the test set

                test_losses = [test_model(xt, yt) for xt, yt in dataset.test(batch_size)]
                test_score  = numpy.mean(test_losses)

                print(('     epoch %i, iter %i, test error of best ' 
                       'model %f %%') % \
                  (epoch, iter, test_score*100.))

        if patience <= iter :
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 ( best_validation_loss * 100., test_score * 100.))
    print ('The code ran for %f minutes' % ((end_time-start_time) / 60.))
    
    return best_validation_loss, test_score, iter*batch_size, (end_time-start_time) / 60.

if __name__ == '__main__':
    log_reg()
    
 
def jobman_log_reg(state, channel):
    print state
    (validation_error, test_error, nb_exemples, time) = log_reg( learning_rate = state.learning_rate, \
                                                                 nb_max_examples = state.nb_max_examples, \
                                                                 dataset=eval(state.dataset), \
                                                                 batch_size  = state.batch_size,\
                                                                 image_size = state.image_size,  \
                                                                 nb_class  = state.nb_class, \
                                                                 patience = state.patience, \
                                                                 patience_increase = state.patience_increase, \
                                                                 improvement_threshold = state.improvement_threshold ) 
                                                                                                    
                                                                                                   
    print state
    state.validation_error = validation_error
    state.test_error = test_error
    state.nb_exemples = nb_exemples
    state.time = time
    return channel.COMPLETE
                                                                
                                      
    
    


