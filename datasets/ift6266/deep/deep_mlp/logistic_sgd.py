import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in,n_out),
                                dtype = theano.config.floatX),
                                name='W')
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                dtype = theano.config.floatX),
                               name='b')

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)

        self.y_pred=T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])


    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    # Load the dataset 
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are 
        # floats it doesn't make sense) therefore instead of returning 
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval




def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='../data/mnist.pkl.gz',
        batch_size = 600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    # compute number of minibatches for training, validation and testing
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

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression( input=x, n_in=28*28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of 
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y) 

    # compiling a Theano function that computes the mistakes that are made by 
    # the model on a minibatch
    test_model = theano.function(inputs = [index], 
            outputs = classifier.errors(y),
            givens={
                x:test_set_x[index*batch_size:(index+1)*batch_size],
                y:test_set_y[index*batch_size:(index+1)*batch_size]})

    validate_model = theano.function( inputs = [index], 
            outputs = classifier.errors(y),
            givens={
                x:valid_set_x[index*batch_size:(index+1)*batch_size],
                y:valid_set_y[index*batch_size:(index+1)*batch_size]})

    # compute the gradient of cost with respect to theta = (W,b) 
    g_W = T.grad(cost = cost, wrt = classifier.W)
    g_b = T.grad(cost = cost, wrt = classifier.b)

    # specify how to update the parameters of the model as a dictionary
    updates ={classifier.W: classifier.W - learning_rate*g_W,\
              classifier.b: classifier.b - learning_rate*g_b}

    # compiling a Theano function `train_model` that returns the cost, but in 
    # the same time updates the parameter of the model based on the rules 
    # defined in `updates`
    train_model = theano.function(inputs = [index], 
            outputs = cost, 
            updates = updates,
            givens={
                x:train_set_x[index*batch_size:(index+1)*batch_size],
                y:train_set_y[index*batch_size:(index+1)*batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience              = 5000  # look as this many examples regardless
    patience_increase     = 2     # wait this much longer when a new best is 
                                  # found
    improvement_threshold = 0.995 # a relative improvement of this much is 
                                  # considered significant
    validation_frequency  = min(n_train_batches, patience/2)  
                                  # go through this many 
                                  # minibatche before checking the network 
                                  # on the validation set; in this case we 
                                  # check every epoch 

    best_params          = None
    best_validation_loss = float('inf')
    test_score           = 0.
    start_time = time.clock()

    done_looping = False 
    epoch = 0  
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
                    test_score  = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best ' 
                       'model %f %%') % \
                        (epoch, minibatch_index+1, n_train_batches,test_score*100.))

            if patience <= iter :
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %  
                 (best_validation_loss * 100., test_score*100.))
    print 'The code run for %d epochs, with %f epochs/sec'%(epoch,1.*epoch/(end_time-start_time))
    print >> sys.stderr, ('The code for file '+os.path.split(__file__)[1]+' ran for %.1fs' % ((end_time-start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()


