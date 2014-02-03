# -*- coding: utf-8 -*-
"""
This tutorial introduces the multilayer perceptron using Theano.  

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermidiate layer, called the hidden layer, that has a nonlinear 
activation function (usually tanh or sigmoid) . One can use many such 
hidden layers making the architecture deep. The tutorial will also tackle 
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" - 
                 Christopher M. Bishop, section 5

TODO: recommended preprocessing, lr ranges, regularization ranges (explain 
      to do lr first, then add regularization)

"""
__docformat__ = 'restructedtext en'

import setup_batches
import pdb
import numpy

import theano
import theano.tensor as T
import time 
import theano.tensor.nnet
import pylearn
import theano,pylearn.version
from pylearn.io import filetensor as ft

data_path = '/data/lisa/data/nist/by_class/'

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model 
    that has one layer or more of hidden units and nonlinear activations. 
    Intermidiate layers usually have as activation function thanh or the 
    sigmoid function  while the top layer is a softamx layer. 
    """



    def __init__(self, input, n_in, n_hidden, n_out,learning_rate, test_subclass):
        """Initialize the parameters for the multilayer perceptron

        :param input: symbolic variable that describes the input of the 
        architecture (one minibatch)

        :param n_in: number of input units, the dimension of the space in 
        which the datapoints lie

        :param n_hidden: number of hidden units 

        :param n_out: number of output units, the dimension of the space in 
        which the labels lie

        """

        # initialize the parameters theta = (W1,b1,W2,b2) ; note that this 
        # example contains only one hidden layer, but one can have as many 
        # layers as he/she wishes, making the network deeper. The only 
        # problem making the network deep this way is during learning, 
        # backpropagation being unable to move the network from the starting
        # point towards; this is where pre-training helps, giving a good 
        # starting point for backpropagation, but more about this in the 
        # other tutorials
        
        # `W1` is initialized with `W1_values` which is uniformely sampled
        # from -6./sqrt(n_in+n_hidden) and 6./sqrt(n_in+n_hidden)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        W1_values = numpy.asarray( numpy.random.uniform( \
              low = -numpy.sqrt(6./(n_in+n_hidden)), \
              high = numpy.sqrt(6./(n_in+n_hidden)), \
              size = (n_in, n_hidden)), dtype = theano.config.floatX)
        # `W2` is initialized with `W2_values` which is uniformely sampled 
        # from -6./sqrt(n_hidden+n_out) and 6./sqrt(n_hidden+n_out)
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        W2_values = numpy.asarray( numpy.random.uniform( 
              low = -numpy.sqrt(6./(n_hidden+n_out)), \
              high= numpy.sqrt(6./(n_hidden+n_out)),\
              size= (n_hidden, n_out)), dtype = theano.config.floatX)

        self.W1 = theano.shared( value = W1_values )
        self.b1 = theano.shared( value = numpy.zeros((n_hidden,), 
                                                dtype= theano.config.floatX))
        self.W2 = theano.shared( value = W2_values )
        self.b2 = theano.shared( value = numpy.zeros((n_out,), 
                                                dtype= theano.config.floatX))

        #include the learning rate in the classifer so
        #we can modify it on the fly when we want
        lr_value=learning_rate
        self.lr=theano.shared(value=lr_value)
        # symbolic expression computing the values of the hidden layer
        self.hidden = T.tanh(T.dot(input, self.W1)+ self.b1)
        
        

        # symbolic expression computing the values of the top layer 
        self.p_y_given_x= T.nnet.softmax(T.dot(self.hidden, self.W2)+self.b2)

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        #self.y_pred = T.argmax( self.p_y_given_x, axis =1)
        #self.y_pred_num = T.argmax( self.p_y_given_x[0:9], axis =1)
        
	self.test_subclass = test_subclass

	#if (self.test_subclass == "u"):
	#  self.y_pred = T.argmax( self.p_y_given_x[10:35], axis =1) + 10
        #elif (self.test_subclass == "l"):
	#  self.y_pred = T.argmax( self.p_y_given_x[35:], axis =1) + 35
        #elif (self.test_subclass == "d"):
	#  self.y_pred = T.argmax( self.p_y_given_x[0:9], axis =1)
        #else:
	self.y_pred = T.argmax( self.p_y_given_x, axis =1)

        # L1 norm ; one regularization option is to enforce L1 norm to 
        # be small 
        self.L1     = abs(self.W1).sum() + abs(self.W2).sum()

        # square of L2 norm ; one regularization option is to enforce 
        # square of L2 norm to be small
        self.L2_sqr = (self.W1**2).sum() + (self.W2**2).sum()



    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|}\mathcal{L} (\theta=\{W,b\}, \mathcal{D}) = 
            \frac{1}{|\mathcal{D}|}\sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D}) 


        :param y: corresponds to a vector that gives for each example the
        :correct label
        """
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


def mlp_full_nist(      verbose = False,\
                        adaptive_lr = 1,\
                        train_data = 'all/all_train_data.ft',\
                        train_labels = 'all/all_train_labels.ft',\
                        test_data = 'all/all_test_data.ft',\
                        test_labels = 'all/all_test_labels.ft',\
                        learning_rate=0.5,\
                        L1_reg = 0.00,\
                        L2_reg = 0.0001,\
                        nb_max_exemples=1000000,\
                        batch_size=20,\
                        nb_hidden = 500,\
                        nb_targets = 26,\
			tau=1e6,\
			main_class="l",\
			start_ratio=1,\
			end_ratio=1):
   
    
    configuration = [learning_rate,nb_max_exemples,nb_hidden,adaptive_lr]
    
    #save initial learning rate if classical adaptive lr is used
    initial_lr=learning_rate
    
    total_validation_error_list = []
    total_train_error_list = []
    learning_rate_list=[]
    best_training_error=float('inf');

    # set up batches
    batches = setup_batches.Batches()
    batches.set_batches(main_class, start_ratio,end_ratio,batch_size,verbose)

    train_batches = batches.get_train_batches()
    test_batches = batches.get_test_batches()
    validation_batches = batches.get_validation_batches()

    ishape     = (32,32) # this is the size of NIST images

    # allocate symbolic variables for the data
    x = T.fmatrix()  # the data is presented as rasterized images
    y = T.lvector()  # the labels are presented as 1D vector of 
                          # [long int] labels

    if verbose==True:
        print 'finished parsing the data'
    # construct the logistic regression class
    classifier = MLP( input=x.reshape((batch_size,32*32)),\
                        n_in=32*32,\
                        n_hidden=nb_hidden,\
                        n_out=nb_targets,\
                        learning_rate=learning_rate,\
			test_subclass=main_class)
                        
                        
   

    # the cost we minimize during training is the negative log likelihood of 
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr 

    # compiling a theano function that computes the mistakes that are made by 
    # the model on a minibatch
    test_model = theano.function([x,y], classifier.errors(y))

    # compute the gradient of cost with respect to theta = (W1, b1, W2, b2) 
    g_W1 = T.grad(cost, classifier.W1)
    g_b1 = T.grad(cost, classifier.b1)
    g_W2 = T.grad(cost, classifier.W2)
    g_b2 = T.grad(cost, classifier.b2)

    # specify how to update the parameters of the model as a dictionary
    updates = \
        { classifier.W1: classifier.W1 - classifier.lr*g_W1 \
        , classifier.b1: classifier.b1 - classifier.lr*g_b1 \
        , classifier.W2: classifier.W2 - classifier.lr*g_W2 \
        , classifier.b2: classifier.b2 - classifier.lr*g_b2 }

    # compiling a theano function `train_model` that returns the cost, but in 
    # the same time updates the parameter of the model based on the rules 
    # defined in `updates`
    train_model = theano.function([x, y], cost, updates = updates )
    n_minibatches        = len(train_batches)

   
   
    
   
   
   #conditions for stopping the adaptation:
   #1) we have reached  nb_max_exemples (this is rounded up to be a multiple of the train size)
   #2) validation error is going up twice in a row(probable overfitting)
   
   # This means we no longer stop on slow convergence as low learning rates stopped
   # too fast. 
   
   # no longer relevant
    patience              =nb_max_exemples/batch_size
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency = n_minibatches/4
   
     

   
    best_params          = None
    best_validation_loss = float('inf')
    best_iter            = 0
    test_score           = 0.
    start_time = time.clock()
    n_iter = nb_max_exemples/batch_size  # nb of max times we are allowed to run through all exemples
    n_iter = n_iter/n_minibatches + 1 #round up
    n_iter=max(1,n_iter) # run at least once on short debug call
    time_n=0 #in unit of exemples
    
    if (main_class == "u"):
      class_offset = 10
    elif (main_class == "l"):
      class_offset = 36
    else:
      class_offset = 0

   
    if verbose == True:
        print 'looping at most %d times through the data set' %n_iter
    for iter in xrange(n_iter* n_minibatches):

        # get epoch and minibatch index
        epoch           = iter / n_minibatches
        minibatch_index =  iter % n_minibatches
        
	
	if adaptive_lr==2:
	    classifier.lr.value = tau*initial_lr/(tau+time_n)
        
        # get the minibatches corresponding to `iter` modulo
        # `len(train_batches)`
        x,y = train_batches[ minibatch_index ]

	y = y - class_offset

        # convert to float
        x_float = x/255.0
        cost_ij = train_model(x_float,y)

        if (iter+1) % validation_frequency == 0: 
            # compute zero-one loss on validation set 
            
            this_validation_loss = 0.
            for x,y in validation_batches:
                # sum up the errors for each minibatch
		y = y - class_offset
                x_float = x/255.0
                this_validation_loss += test_model(x_float,y)
            # get the average by dividing with the number of minibatches
            this_validation_loss /= len(validation_batches)
            #save the validation loss
            total_validation_error_list.append(this_validation_loss)
            
            #get the training error rate
            this_train_loss=0
            for x,y in train_batches:
                # sum up the errors for each minibatch
		y = y - class_offset
                x_float = x/255.0
                this_train_loss += test_model(x_float,y)
            # get the average by dividing with the number of minibatches
            this_train_loss /= len(train_batches)
            #save the validation loss
            total_train_error_list.append(this_train_loss)
            if(this_train_loss<best_training_error):
                best_training_error=this_train_loss
                
            if verbose == True:
                print('epoch %i, minibatch %i/%i, validation error %f, training error %f %%' % \
                    (epoch, minibatch_index+1, n_minibatches, \
                        this_validation_loss*100.,this_train_loss*100))
		print 'learning rate = %f' %classifier.lr.value
		print 'time  = %i' %time_n
                        
                        
            #save the learning rate
            learning_rate_list.append(classifier.lr.value)


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter
                # reset patience if we are going down again
                # so we continue exploring
                patience=nb_max_exemples/batch_size
                # test it on the test set
                test_score = 0.
                for x,y in test_batches:
		    y = y - class_offset
                    x_float=x/255.0
                    test_score += test_model(x_float,y)
                test_score /= len(test_batches)
                if verbose == True:
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                        'model %f %%') % 
                                (epoch, minibatch_index+1, n_minibatches,
                                test_score*100.))
                                
            # if the validation error is going up, we are overfitting (or oscillating)
            # stop converging but run at least to next validation
            # to check overfitting or ocsillation
            # the saved weights of the model will be a bit off in that case
            elif this_validation_loss >= best_validation_loss:
                #calculate the test error at this point and exit
                # test it on the test set
                # however, if adaptive_lr is true, try reducing the lr to
                # get us out of an oscilliation
                if adaptive_lr==1:
                    classifier.lr.value=classifier.lr.value/2.0

                test_score = 0.
                #cap the patience so we are allowed one more validation error
                #calculation before aborting
                patience = iter+validation_frequency+1
                for x,y in test_batches:
		    y = y - class_offset
                    x_float=x/255.0
                    test_score += test_model(x_float,y)
                test_score /= len(test_batches)
                if verbose == True:
                    print ' validation error is going up, possibly stopping soon'
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                        'model %f %%') % 
                                (epoch, minibatch_index+1, n_minibatches,
                                test_score*100.))
                                
                


        if iter>patience:
            print 'we have diverged'
            break


    	time_n= time_n + batch_size
    end_time = time.clock()
    if verbose == True:
        print(('Optimization complete. Best validation score of %f %% '
            'obtained at iteration %i, with test performance %f %%') %  
                    (best_validation_loss * 100., best_iter, test_score*100.))
        print ('The code ran for %f minutes' % ((end_time-start_time)/60.))
        print iter
        
    #save the model and the weights
    numpy.savez('model.npy', config=configuration, W1=classifier.W1.value,W2=classifier.W2.value, b1=classifier.b1.value,b2=classifier.b2.value)
    numpy.savez('results.npy',config=configuration,total_train_error_list=total_train_error_list,total_validation_error_list=total_validation_error_list,\
    learning_rate_list=learning_rate_list)
    
    return (best_training_error*100.0,best_validation_loss * 100.,test_score*100.,best_iter*batch_size,(end_time-start_time)/60)


if __name__ == '__main__':
    mlp_full_nist(True)

def jobman_mlp_full_nist(state,channel):
    (train_error,validation_error,test_error,nb_exemples,time)=mlp_full_nist(learning_rate=state.learning_rate,\
                                                                nb_hidden=state.nb_hidden,\
                                                                main_class=state.main_class,\
								start_ratio=state.ratio,\
								end_ratio=state.ratio)
    state.train_error=train_error
    state.validation_error=validation_error
    state.test_error=test_error
    state.nb_exemples=nb_exemples
    state.time=time
    return channel.COMPLETE
                                                                
                                                                