__docformat__ = 'restructedtext en'

import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation = T.tanh):
        print "Creating HiddenLayer with params"
        print locals()

        self.input = input

        W_values = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_in+n_out)),
                high = numpy.sqrt(6./(n_in+n_out)),
                size = (n_in, n_out)), dtype = theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value = W_values, name ='W')

        b_values = numpy.zeros((n_out,), dtype= theano.config.floatX)
        self.b = theano.shared(value= b_values, name ='b')

        self.output = activation(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden_layers, n_hidden, n_out):
        print "Creating MLP with params"
        print locals()

        self.input = input

        self.hiddenLayers = []

        last_input = input
        last_n_out = n_in
        for i in range(n_hidden_layers):
            self.hiddenLayers.append(\
                    HiddenLayer(rng = rng, input = last_input, 
                                             n_in = last_n_out,
                                             n_out = n_hidden,
                                             activation = T.tanh))
            last_input = self.hiddenLayers[-1].output
            last_n_out = n_hidden

        self.logRegressionLayer = LogisticRegression( 
                                    input = self.hiddenLayers[-1].output,
                                    n_in  = n_hidden,
                                    n_out = n_out)

        self.L1 = abs(self.logRegressionLayer.W).sum()
        for h in self.hiddenLayers:
            self.L1 += abs(h.W).sum()

        self.L2_sqr = (self.logRegressionLayer.W**2).sum()
        for h in self.hiddenLayers:
            self.L2_sqr += (h.W**2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors

        self.params = []
        for hl in self.hiddenLayers:
            self.params += hl.params
        self.params += self.logRegressionLayer.params


def test_mlp( learning_rate=0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs=1000,
            dataset = '../data/mnist.pkl.gz', batch_size = 20):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    n_train_batches = train_set_x.value.shape[0] / batch_size
    n_valid_batches = valid_set_x.value.shape[0] / batch_size
    n_test_batches  = test_set_x.value.shape[0]  / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ###################### 
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images
    y     = T.ivector('y') # the labels are presented as 1D vector of 
                           # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP( rng = rng, input=x, n_in=28*28, n_hidden = 500, n_out=10)

    # the cost we minimize during training is the negative log likelihood of 
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr 

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(inputs = [index], 
            outputs = classifier.errors(y),
            givens={
                x:test_set_x[index*batch_size:(index+1)*batch_size],
                y:test_set_y[index*batch_size:(index+1)*batch_size]})

    validate_model = theano.function(inputs = [index], 
            outputs = classifier.errors(y),
            givens={
                x:valid_set_x[index*batch_size:(index+1)*batch_size],
                y:valid_set_y[index*batch_size:(index+1)*batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in classifier.params:
        gparam  = T.grad(cost, param)
        gparams.append(gparam)


    # specify how to update the parameters of the model as a dictionary
    updates = {}
    # given two list the zip A = [ a1,a2,a3,a4] and B = [b1,b2,b3,b4] of 
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists : 
    #    C = [ (a1,b1), (a2,b2), (a3,b3) , (a4,b4) ] 
    for param, gparam in zip(classifier.params, gparams):
        updates[param] = param - learning_rate*gparam

    # compiling a Theano function `train_model` that returns the cost, but  
    # in the same time updates the parameter of the model based on the rules 
    # defined in `updates`
    train_model =theano.function( inputs = [index], outputs = cost, 
            updates = updates,
            givens={
                x:train_set_x[index*batch_size:(index+1)*batch_size],
                y:train_set_y[index*batch_size:(index+1)*batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience              = 10000 # look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_batches,patience/2)  
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 


    best_params          = None
    best_validation_loss = float('inf')
    best_iter            = 0
    test_score           = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in xrange(n_train_batches):

        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = epoch * n_train_batches + minibatch_index

        if (iter+1) % validation_frequency == 0: 
            # compute zero-one loss on validation set 
            validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)

            print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                 (epoch, minibatch_index+1,n_train_batches, \
                  this_validation_loss*100.))


            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold :
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                # test it on the test set

                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)

          
