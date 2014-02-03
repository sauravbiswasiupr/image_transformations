import time
import sys, os

from ift6266.utils.seriestables import *

default_series = {
    'train_error' : DummySeries(),
    'valid_error' : DummySeries(),
    'test_error' : DummySeries()
    }

def sgd_opt(train, valid, test, training_epochs=10000, patience=10000,
            patience_increase=2., improvement_threshold=0.995, net=None,
            validation_frequency=None, series=default_series):

    if validation_frequency is None:
        validation_frequency = patience/2
 
    start_time = time.clock()

    best_params = None
    best_validation_loss = float('inf')
    test_score = 0.

    start_time = time.clock()
 
    for epoch in xrange(1, training_epochs+1):
        series['train_error'].append((epoch,), train())

        if epoch % validation_frequency == 0:
            this_validation_loss = valid()
            series['valid_error'].append((epoch,), this_validation_loss*100.)
            print('epoch %i, validation error %f %%' % \
                   (epoch, this_validation_loss*100.))
            
            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
 
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * \
                       improvement_threshold :
                    patience = max(patience, epoch * patience_increase)
                
                # save best validation score and epoch number
                best_validation_loss = this_validation_loss
                best_epoch = epoch
                
                # test it on the test set
                test_score = test()
                series['test_error'].append((epoch,), test_score*100.)
                print((' epoch %i, test error of best model %f %%') %
                      (epoch, test_score*100.))
                if net is not None:
                    net.save('best.net.new')
                    os.rename('best.net.new', 'best.net')
                
        if patience <= epoch:
            break
    
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score*100.))
    print ('The code ran for %f minutes' % ((end_time-start_time)/60.))

    return best_validation_loss, test_score
