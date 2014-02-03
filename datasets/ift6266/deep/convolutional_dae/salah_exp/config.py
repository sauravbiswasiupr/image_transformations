'''
These are parameters used by nist_sda.py. They'll end up as globals in there.

Rename this file to config.py and configure as needed.
DON'T add the renamed file to the repository, as others might use it
without realizing it, with dire consequences.
'''

# Set this to True when you want to run cluster tests, ie. you want
# to run on the cluster, many jobs, but want to reduce the training
# set size and the number of epochs, so you know everything runs
# fine on the cluster.
# Set this PRIOR to inserting your test jobs in the DB.
TEST_CONFIG = False

NIST_ALL_LOCATION = '/data/lisa/data/nist/by_class/all'
NIST_ALL_TRAIN_SIZE = 649081
# valid et test =82587 82587 

# change "sandbox" when you're ready
JOBDB = 'postgres://ift6266h10@gershwin/ift6266h10_db/rifaisal_csda'
EXPERIMENT_PATH = "ift6266.deep.convolutional_dae.salah_exp.nist_csda.jobman_entrypoint"

##Pour lancer des travaux sur le cluster: (il faut etre ou se trouve les fichiers)
##python nist_sda.py jobman_insert
##dbidispatch --condor --repeat_jobs=2 jobman sql 'postgres://ift6266h10@gershwin/ift6266h10_db/pannetis_finetuningSDA0' .  #C'est le path dans config.py

# reduce training set to that many examples
REDUCE_TRAIN_TO = None
# that's a max, it usually doesn't get to that point
MAX_FINETUNING_EPOCHS = 1000
# number of minibatches before taking means for valid error etc.
REDUCE_EVERY = 100
#Set the finetune dataset
FINETUNE_SET=1
#Set the pretrain dataset used. 0: NIST, 1:P07
PRETRAIN_CHOICE=1


if TEST_CONFIG:
    REDUCE_TRAIN_TO = 1000
    MAX_FINETUNING_EPOCHS = 2
    REDUCE_EVERY = 10


# This is to configure insertion of jobs on the cluster.
# Possible values the hyperparameters can take. These are then
# combined with produit_cartesien_jobs so we get a list of all
# possible combinations, each one resulting in a job inserted
# in the jobman DB.


JOB_VALS = {'pretraining_lr': [0.01],#, 0.001],#, 0.0001],
        'pretraining_epochs_per_layer': [10],
        'kernels' : [[[52,5,5], [32,3,3]], [[52,7,7], [52,3,3]]],
        'mlp_size' : [[1000],[500]],
        'imgshp' : [[32,32]],
        'max_pool_layers' : [[[2,2],[2,2]]],
        'corruption_levels': [[0.2,0.1]],
        'minibatch_size': [100],
        'max_finetuning_epochs':[MAX_FINETUNING_EPOCHS],
        'max_finetuning_epochs_P07':[1000],
        'finetuning_lr':[0.1,0.01], #0.001 was very bad, so we leave it out
        'num_hidden_layers':[2],
        'finetune_set':[1],
        'pretrain_choice':[1]
        }

DEFAULT_HP_NIST = {'pretraining_lr': 0.01,
        'pretraining_epochs_per_layer': 1,
        'kernels' : [[4,5,5], [2,3,3]],
        'mlp_size' : [10],
        'imgshp' : [32,32],
        'max_pool_layers' : [[2,2],[2,2]],
        'corruption_levels': [0.1,0.2],
        'minibatch_size': 20,
        'max_finetuning_epochs':MAX_FINETUNING_EPOCHS,
        'max_finetuning_epochs_P07':1000,
        'finetuning_lr':0.1, #0.001 was very bad, so we leave it out
        'num_hidden_layers':2,
        'finetune_set':1,
        'pretrain_choice':1,
        #'reduce_train_to':1000,
        }

                    
                    
##[pannetis@ceylon test]$ python nist_sda.py test_jobman_entrypoint
##WARNING: untracked file /u/pannetis/IFT6266/ift6266/deep/stacked_dae/v_sylvain/TMP_DBI/configobj.py
##WARNING: untracked file /u/pannetis/IFT6266/ift6266/deep/stacked_dae/v_sylvain/TMP_DBI/utils.py
##WARNING: untracked file /u/pannetis/IFT6266/ift6266/deep/stacked_dae/v_sylvain/config.py
##WARNING: untracked file /u/pannetis/IFT6266/ift6266/deep/stacked_dae/v_sylvain/config2.py
##Creating optimizer with state,  DD{'reduce_train_to': 11000, 'pretraining_epochs_per_layer': 2, 'hidden_layers_sizes': 300, 'num_hidden_layers': 2, 'corruption_levels': 0.20000000000000001, 'finetuning_lr': 0.10000000000000001, 'pretrain_choice': 0, 'max_finetuning_epochs': 2, 'version_pylearn': '08b37147dec1', 'finetune_set': -1, 'pretraining_lr': 0.10000000000000001, 'version_ift6266': 'a6b6b1140de9', 'version_theano': 'fb6c3a06cb65', 'minibatch_size': 20}
##SdaSgdOptimizer, max_minibatches = 11000
##C##n_outs 62
##pretrain_lr 0.1
##finetune_lr 0.1
##----
##
##pretraining with NIST
##
##STARTING PRETRAINING, time =  2010-03-29 15:07:43.945981
##Pre-training layer 0, epoch 0, cost  113.562562494
##Pre-training layer 0, epoch 1, cost  113.410032944
##Pre-training layer 1, epoch 0, cost  98.4539954687
##Pre-training layer 1, epoch 1, cost  97.8658966686
##Pretraining took 9.011333 minutes
##
##SERIE OF 3 DIFFERENT FINETUNINGS
##
##
##finetune with NIST
##
##
##STARTING FINETUNING, time =  2010-03-29 15:16:46.512235
##epoch 1, minibatch 4999, validation error on P07 : 29.511250 %
##     epoch 1, minibatch 4999, test error on dataset NIST  (train data) of best model 40.408509 %
##     epoch 1, minibatch 4999, test error on dataset P07 of best model 96.700000 %
##epoch 1, minibatch 9999, validation error on P07 : 25.560000 %
##     epoch 1, minibatch 9999, test error on dataset NIST  (train data) of best model 34.778969 %
##     epoch 1, minibatch 9999, test error on dataset P07 of best model 97.037500 %
##
##Optimization complete with best validation score of 25.560000 %,with test performance 34.778969 % on dataset NIST 
##The test score on the P07 dataset is 97.037500
##The finetuning ran for 3.281833 minutes
##
##
##finetune with P07
##
##
##STARTING FINETUNING, time =  2010-03-29 15:20:06.346009
##epoch 1, minibatch 4999, validation error on NIST : 65.226250 %
##     epoch 1, minibatch 4999, test error on dataset P07  (train data) of best model 84.465000 %
##     epoch 1, minibatch 4999, test error on dataset NIST of best model 65.965237 %
##epoch 1, minibatch 9999, validation error on NIST : 58.745000 %
##     epoch 1, minibatch 9999, test error on dataset P07  (train data) of best model 80.405000 %
##     epoch 1, minibatch 9999, test error on dataset NIST of best model 61.341923 %
##
##Optimization complete with best validation score of 58.745000 %,with test performance 80.405000 % on dataset P07 
##The test score on the NIST dataset is 61.341923
##The finetuning ran for 3.299500 minutes
##
##
##finetune with NIST (done earlier) followed by P07 (written here)
##
##
##STARTING FINETUNING, time =  2010-03-29 15:23:27.947374
##epoch 1, minibatch 4999, validation error on NIST : 83.975000 %
##     epoch 1, minibatch 4999, test error on dataset P07  (train data) of best model 83.872500 %
##     epoch 1, minibatch 4999, test error on dataset NIST of best model 43.170010 %
##epoch 1, minibatch 9999, validation error on NIST : 79.775000 %
##     epoch 1, minibatch 9999, test error on dataset P07  (train data) of best model 80.971250 %
##     epoch 1, minibatch 9999, test error on dataset NIST of best model 49.017468 %
##
##Optimization complete with best validation score of 79.775000 %,with test performance 80.971250 % on dataset P07 
##The test score on the NIST dataset is 49.017468
##The finetuning ran for 2.851500 minutes
##
##
##finetune with NIST only on the logistic regression on top.
##        All hidden units output are input of the logistic regression
##
##
##STARTING FINETUNING, time =  2010-03-29 15:26:21.430557
##epoch 1, minibatch 4999, validation error on P07 : 95.223750 %
##     epoch 1, minibatch 4999, test error on dataset NIST  (train data) of best model 93.268765 %
##     epoch 1, minibatch 4999, test error on dataset P07 of best model 96.535000 %
##epoch 1, minibatch 9999, validation error on P07 : 95.223750 %
##
##Optimization complete with best validation score of 95.223750 %,with test performance 93.268765 % on dataset NIST 
##The test score on the P07 dataset is 96.535000
##The finetuning ran for 2.013167 minutes
##Closing remaining open files: /u/pannetis/IFT6266/test/series.h5... done
##[pannetis@ceylon test]$ 



