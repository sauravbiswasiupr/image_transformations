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

import sys
import pdb
import numpy
import pylab
import theano
import theano.tensor as T
import time 
import theano.tensor.nnet
import pylearn
import theano,pylearn.version,ift6266
from pylearn.io import filetensor as ft
from ift6266 import datasets

data_path = '/data/lisa/data/nist/by_class/'

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model 
    that has one layer or more of hidden units and nonlinear activations. 
    Intermidiate layers usually have as activation function thanh or the 
    sigmoid function  while the top layer is a softamx layer. 
    """



    def __init__(self, input, n_in, n_hidden, n_out,learning_rate,detection_mode):
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
        if(detection_mode==0):
            self.p_y_given_x= T.nnet.softmax(T.dot(self.hidden, self.W2)+self.b2)
        else:
            self.p_y_given_x= T.nnet.sigmoid(T.dot(self.hidden, self.W2)+self.b2)
            
        
        
       # self.y_out_sig= T.sigmoid(T.dot(self.hidden, self.W2)+self.b2)

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred = T.argmax( self.p_y_given_x, axis =1)
        
       # self.y_pred_sig = T.argmax( self.y_out_sig, axis =1)
        
        
        
        
        
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
    
    
    def cross_entropy(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]+T.sum(T.log(1-self.p_y_given_x), axis=1)-T.log(1-self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    




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

def mlp_get_nist_error(model_name='/u/mullerx/ift6266h10_sandbox_db/xvm_final_lr1_p073/8/best_model.npy.npz',
                  data_set=0):
    


    

    

    # load the data set and create an mlp based on the dimensions of the model
    model=numpy.load(model_name)
    W1=model['W1']
    W2=model['W2']
    b1=model['b1']
    b2=model['b2']
    
    total_error_count=0.0
    total_exemple_count=0.0
    
    nb_error_count=0.0
    nb_exemple_count=0.0
    
    char_error_count=0.0
    char_exemple_count=0.0
    
    min_error_count=0.0
    min_exemple_count=0.0
    
    maj_error_count=0.0
    maj_exemple_count=0.0
    
    vtotal_error_count=0.0
    vtotal_exemple_count=0.0
    
    vnb_error_count=0.0
    vnb_exemple_count=0.0
    
    vchar_error_count=0.0
    vchar_exemple_count=0.0
    
    vmin_error_count=0.0
    vmin_exemple_count=0.0
    
    vmaj_error_count=0.0
    vmaj_exemple_count=0.0
    
    nbc_error_count=0.0
    vnbc_error_count=0.0
    
    

    if data_set==0:
        print 'using nist'
    	dataset=datasets.nist_all()
    elif data_set==1:
        print 'using p07'
        dataset=datasets.nist_P07()
    elif data_set==2:
        print 'using pnist'
        dataset=datasets.PNIST07()
        
   



    #get the test error
    #use a batch size of 1 so we can get the sub-class error
    #without messing with matrices (will be upgraded later)
    test_score=0
    temp=0
    for xt,yt in dataset.test(1):
        
        total_exemple_count = total_exemple_count +1
        #get activation for layer 1
        a0=numpy.dot(numpy.transpose(W1),numpy.transpose(xt[0])) + b1
        #add non linear function to layer 1 activation
        a0_out=numpy.tanh(a0)
        
        #get activation for output layer
        a1= numpy.dot(numpy.transpose(W2),a0_out) + b2
        #add non linear function for output activation (softmax)
        a1_exp = numpy.exp(a1)
        sum_a1=numpy.sum(a1_exp)
        a1_out=a1_exp/sum_a1
        
        predicted_class=numpy.argmax(a1_out)
        wanted_class=yt[0]
        if(predicted_class!=wanted_class):
            total_error_count = total_error_count +1
        
        
        if(not(predicted_class==wanted_class or ( (((predicted_class+26)==wanted_class) or ((predicted_class-26)==wanted_class)) and wanted_class>9)   )):
            nbc_error_count = nbc_error_count +1
               
               
        #treat digit error
        if(wanted_class<10):
            nb_exemple_count=nb_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[0:10])
            if(predicted_class!=wanted_class):
                nb_error_count = nb_error_count +1
                
        if(wanted_class>9):
            char_exemple_count=char_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[10:62])+10
            if((predicted_class!=wanted_class) and ((predicted_class+26)!=wanted_class) and ((predicted_class-26)!=wanted_class)):
               char_error_count = char_error_count +1
               
        #minuscule
        if(wanted_class>9 and wanted_class<36):
            maj_exemple_count=maj_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[10:35])+10
            if(predicted_class!=wanted_class):
                maj_error_count = maj_error_count +1
        #majuscule
        if(wanted_class>35):
            min_exemple_count=min_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[36:62])+36
            if(predicted_class!=wanted_class):
                min_error_count = min_error_count +1
            
            
            
    vtest_score=0
    vtemp=0
    for xt,yt in dataset.valid(1):
        
        vtotal_exemple_count = vtotal_exemple_count +1
        #get activation for layer 1
        a0=numpy.dot(numpy.transpose(W1),numpy.transpose(xt[0])) + b1
        #add non linear function to layer 1 activation
        a0_out=numpy.tanh(a0)
        
        #get activation for output layer
        a1= numpy.dot(numpy.transpose(W2),a0_out) + b2
        #add non linear function for output activation (softmax)
        a1_exp = numpy.exp(a1)
        sum_a1=numpy.sum(a1_exp)
        a1_out=a1_exp/sum_a1
        
        predicted_class=numpy.argmax(a1_out)
        wanted_class=yt[0]
        if(predicted_class!=wanted_class):
            vtotal_error_count = vtotal_error_count +1
            
        if(not(predicted_class==wanted_class or ( (((predicted_class+26)==wanted_class) or ((predicted_class-26)==wanted_class)) and wanted_class>9)   )):
            vnbc_error_count = nbc_error_count +1
            
        #treat digit error
        if(wanted_class<10):
            vnb_exemple_count=vnb_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[0:10])
            if(predicted_class!=wanted_class):
                vnb_error_count = vnb_error_count +1
                
        if(wanted_class>9):
            vchar_exemple_count=vchar_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[10:62])+10
            if((predicted_class!=wanted_class) and ((predicted_class+26)!=wanted_class) and ((predicted_class-26)!=wanted_class)):
               vchar_error_count = vchar_error_count +1
               
        #minuscule
        if(wanted_class>9 and wanted_class<36):
            vmaj_exemple_count=vmaj_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[10:35])+10
            if(predicted_class!=wanted_class):
                vmaj_error_count = vmaj_error_count +1
        #majuscule
        if(wanted_class>35):
            vmin_exemple_count=vmin_exemple_count + 1
            predicted_class=numpy.argmax(a1_out[36:62])+36
            if(predicted_class!=wanted_class):
                vmin_error_count = vmin_error_count +1
            

    print (('total error = %f') % ((total_error_count/total_exemple_count)*100.0))
    print (('number error = %f') % ((nb_error_count/nb_exemple_count)*100.0))
    print (('char error = %f') % ((char_error_count/char_exemple_count)*100.0))
    print (('min error = %f') % ((min_error_count/min_exemple_count)*100.0))
    print (('maj error = %f') % ((maj_error_count/maj_exemple_count)*100.0))
    print (('36 error = %f') % ((nbc_error_count/total_exemple_count)*100.0))
    
    print (('valid total error = %f') % ((vtotal_error_count/vtotal_exemple_count)*100.0))
    print (('valid number error = %f') % ((vnb_error_count/vnb_exemple_count)*100.0))
    print (('valid char error = %f') % ((vchar_error_count/vchar_exemple_count)*100.0))
    print (('valid min error = %f') % ((vmin_error_count/vmin_exemple_count)*100.0))
    print (('valid maj error = %f') % ((vmaj_error_count/vmaj_exemple_count)*100.0))
    print (('valid 36 error = %f') % ((vnbc_error_count/vtotal_exemple_count)*100.0))
    
    print (('num total = %d,%d') % (total_exemple_count,total_error_count))
    print (('num nb = %d,%d') % (nb_exemple_count,nb_error_count))
    print (('num min = %d,%d') % (min_exemple_count,min_error_count))
    print (('num maj = %d,%d') % (maj_exemple_count,maj_error_count))
    print (('num char = %d,%d') % (char_exemple_count,char_error_count))
    
    
    
    total_error_count/=total_exemple_count
    nb_error_count/=nb_exemple_count
    char_error_count/=char_exemple_count
    min_error_count/=min_exemple_count
    maj_error_count/=maj_exemple_count
    nbc_error_count/=total_exemple_count
    
    vtotal_error_count/=vtotal_exemple_count
    vnb_error_count/=vnb_exemple_count
    vchar_error_count/=vchar_exemple_count
    vmin_error_count/=vmin_exemple_count
    vmaj_error_count/=vmaj_exemple_count
    vnbc_error_count/=vtotal_exemple_count
    
    
    
    return (total_error_count,nb_error_count,char_error_count,min_error_count,maj_error_count,nbc_error_count,\
            vtotal_error_count,vnb_error_count,vchar_error_count,vmin_error_count,vmaj_error_count,vnbc_error_count)
            
def jobman_get_error(state,channel):
    (all_t_error,nb_t_error,char_t_error,min_t_error,maj_t_error,nbc_t_error,
     all_v_error,nb_v_error,char_v_error,min_v_error,maj_v_error,nbc_v_error)=mlp_get_nist_error(data_set=state.data_set,\
                                                                                     model_name=state.model_name)
    
    state.all_t_error=all_t_error*100.0
    state.nb_t_error=nb_t_error*100.0
    state.char_t_error=char_t_error*100.0
    state.min_t_error=min_t_error*100.0
    state.maj_t_error=maj_t_error*100.0
    state.nbc_t_error=nbc_t_error*100.0
    
    state.all_v_error=all_v_error*100.0
    state.nb_v_error=nb_v_error*100.0
    state.char_v_error=char_v_error*100.0
    state.min_v_error=min_v_error*100.0
    state.maj_v_error=maj_v_error*100.0
    state.nbc_v_error=nbc_v_error*100.0
    
    pylearn.version.record_versions(state,[theano,ift6266,pylearn])
    return channel.COMPLETE





def mlp_full_nist(      verbose = 1,\
                        adaptive_lr = 0,\
                        data_set=0,\
                        learning_rate=0.01,\
                        L1_reg = 0.00,\
                        L2_reg = 0.0001,\
                        nb_max_exemples=1000000,\
                        batch_size=20,\
                        nb_hidden = 30,\
                        nb_targets = 62,
                        tau=1e6,\
                        lr_t2_factor=0.5,\
                        init_model=0,\
                        channel=0,\
                        detection_mode=0):
   
    
    if channel!=0:
        channel.save()
    configuration = [learning_rate,nb_max_exemples,nb_hidden,adaptive_lr]
    
    #save initial learning rate if classical adaptive lr is used
    initial_lr=learning_rate
    max_div_count=1000
    optimal_test_error=0
    
    
    total_validation_error_list = []
    total_train_error_list = []
    learning_rate_list=[]
    best_training_error=float('inf');
    divergence_flag_list=[]
    
    if data_set==0:
        print 'using nist'
    	dataset=datasets.nist_all()
    elif data_set==1:
        print 'using p07'
        dataset=datasets.nist_P07()
    elif data_set==2:
        print 'using pnist'
        dataset=datasets.PNIST07()
    
    
    

    ishape     = (32,32) # this is the size of NIST images

    # allocate symbolic variables for the data
    x = T.fmatrix()  # the data is presented as rasterized images
    y = T.lvector()  # the labels are presented as 1D vector of 
                          # [long int] labels

    
    # construct the logistic regression class
    classifier = MLP( input=x,\
                        n_in=32*32,\
                        n_hidden=nb_hidden,\
                        n_out=nb_targets,
                        learning_rate=learning_rate,
                        detection_mode=detection_mode)
                        
                        
    # check if we want to initialise the weights with a previously calculated model
    # dimensions must be consistent between old model and current configuration!!!!!! (nb_hidden and nb_targets)
    if init_model!=0:
        print 'using old model'
        print init_model
        old_model=numpy.load(init_model)
        classifier.W1.value=old_model['W1']
        classifier.W2.value=old_model['W2']
        classifier.b1.value=old_model['b1']
        classifier.b2.value=old_model['b2']
   

    # the cost we minimize during training is the negative log likelihood of 
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    if(detection_mode==0):
        cost = classifier.negative_log_likelihood(y) \
         + L1_reg * classifier.L1 \
         + L2_reg * classifier.L2_sqr 
    else:
        cost = classifier.cross_entropy(y) \
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
    
    
   

   
   
    
   
   
   #conditions for stopping the adaptation:
   #1) we have reached  nb_max_exemples (this is rounded up to be a multiple of the train size so we always do at least 1 epoch)
   #2) validation error is going up twice in a row(probable overfitting)
   
   # This means we no longer stop on slow convergence as low learning rates stopped
   # too fast but instead we will wait for the valid error going up 3 times in a row
   # We save the curb of the validation error so we can always go back to check on it 
   # and we save the absolute best model anyway, so we might as well explore
   # a bit when diverging
   
    #approximate number of samples in the nist training set
    #this is just to have a validation frequency
    #roughly proportionnal to the original nist training set
    n_minibatches        = 650000/batch_size
    
    
    patience              =2*nb_max_exemples/batch_size #in units of minibatch
    validation_frequency = n_minibatches/4
   
     

   
    
    best_validation_loss = float('inf')
    best_iter            = 0
    test_score           = 0.
    start_time = time.clock()
    time_n=0 #in unit of exemples
    minibatch_index=0
    epoch=0
    temp=0
    divergence_flag=0
    
    
    
    
    print 'starting training'
    sys.stdout.flush()
    while(minibatch_index*batch_size<nb_max_exemples):
        
        for x, y in dataset.train(batch_size):

            #if we are using the classic learning rate deacay, adjust it before training of current mini-batch
            if adaptive_lr==2:
                    classifier.lr.value = tau*initial_lr/(tau+time_n)
        
            
            #train model
            cost_ij = train_model(x,y)
            if (minibatch_index) % validation_frequency == 0: 
                #save the current learning rate
                learning_rate_list.append(classifier.lr.value)
                divergence_flag_list.append(divergence_flag)

                
                
                # compute the validation error
                this_validation_loss = 0.
                temp=0
                for xv,yv in dataset.valid(1):
                    # sum up the errors for each minibatch
                    this_validation_loss += test_model(xv,yv)
                    temp=temp+1
                # get the average by dividing with the number of minibatches
                this_validation_loss /= temp
                #save the validation loss
                total_validation_error_list.append(this_validation_loss)
                
		print(('epoch %i, minibatch %i, learning rate %f current validation error %f ') % 
			(epoch, minibatch_index+1,classifier.lr.value,
			this_validation_loss*100.))
		sys.stdout.flush()
				
		#save temp results to check during training
                numpy.savez('temp_results.npy',config=configuration,total_validation_error_list=total_validation_error_list,\
                learning_rate_list=learning_rate_list, divergence_flag_list=divergence_flag_list)
    
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = minibatch_index
                    #reset divergence flag
                    divergence_flag=0
                    
                    #save the best model. Overwrite the current saved best model so
                    #we only keep the best
                    numpy.savez('best_model.npy', config=configuration, W1=classifier.W1.value, W2=classifier.W2.value, b1=classifier.b1.value,\
                    b2=classifier.b2.value, minibatch_index=minibatch_index)

                    # test it on the test set
                    test_score = 0.
                    temp =0
                    for xt,yt in dataset.test(batch_size):
                        test_score += test_model(xt,yt)
                        temp = temp+1
                    test_score /= temp
                    
		    print(('epoch %i, minibatch %i, test error of best '
			'model %f %%') % 
				(epoch, minibatch_index+1,
				test_score*100.))
                    sys.stdout.flush()
                    optimal_test_error=test_score
                                    
                # if the validation error is going up, we are overfitting (or oscillating)
                # check if we are allowed to continue and if we will adjust the learning rate
                elif this_validation_loss >= best_validation_loss:
                   
                    
                    # In non-classic learning rate decay, we modify the weight only when
                    # validation error is going up
                    if adaptive_lr==1:
                        classifier.lr.value=classifier.lr.value*lr_t2_factor
                           
                   
                    #cap the patience so we are allowed to diverge max_div_count times
                    #if we are going up max_div_count in a row, we will stop immediatelty by modifying the patience
                    divergence_flag = divergence_flag +1
                    
                    
                    #calculate the test error at this point and exit
                    # test it on the test set
                    test_score = 0.
                    temp=0
                    for xt,yt in dataset.test(batch_size):
                        test_score += test_model(xt,yt)
                        temp=temp+1
                    test_score /= temp
                    
                    print ' validation error is going up, possibly stopping soon'
                    print(('     epoch %i, minibatch %i, test error of best '
                        'model %f %%') % 
                                (epoch, minibatch_index+1,
                                test_score*100.))
                    sys.stdout.flush()
                                    
                    
    
            # check early stop condition
            if divergence_flag==max_div_count:
                minibatch_index=nb_max_exemples
                print 'we have diverged, early stopping kicks in'
                break
            
            #check if we have seen enough exemples
            #force one epoch at least
            if epoch>0 and minibatch_index*batch_size>nb_max_exemples:
                break


                       
    
    
            time_n= time_n + batch_size
            minibatch_index =  minibatch_index + 1
            
        # we have finished looping through the training set
        epoch = epoch+1
    end_time = time.clock()
   
    print(('Optimization complete. Best validation score of %f %% '
        'obtained at iteration %i, with test performance %f %%') %  
                (best_validation_loss * 100., best_iter, test_score*100.))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))
    print minibatch_index
    sys.stdout.flush()
        
    #save the model and the weights
    numpy.savez('model.npy', config=configuration, W1=classifier.W1.value,W2=classifier.W2.value, b1=classifier.b1.value,b2=classifier.b2.value)
    numpy.savez('results.npy',config=configuration,total_train_error_list=total_train_error_list,total_validation_error_list=total_validation_error_list,\
    learning_rate_list=learning_rate_list, divergence_flag_list=divergence_flag_list)
    
    return (best_training_error*100.0,best_validation_loss * 100.,optimal_test_error*100.,best_iter*batch_size,(end_time-start_time)/60)


if __name__ == '__main__':
    mlp_full_mnist()

def jobman_mlp_full_nist(state,channel):
    (train_error,validation_error,test_error,nb_exemples,time)=mlp_full_nist(learning_rate=state.learning_rate,\
										nb_max_exemples=state.nb_max_exemples,\
										nb_hidden=state.nb_hidden,\
										adaptive_lr=state.adaptive_lr,\
										tau=state.tau,\
										verbose = state.verbose,\
										lr_t2_factor=state.lr_t2_factor,
                                        data_set=state.data_set,
                                        init_model=state.init_model,
                                        detection_mode = state.detection_mode,\
                                        channel=channel)
    state.train_error=train_error
    state.validation_error=validation_error
    state.test_error=test_error
    state.nb_exemples=nb_exemples
    state.time=time
    pylearn.version.record_versions(state,[theano,ift6266,pylearn])
    return channel.COMPLETE
                                                                
                                                                