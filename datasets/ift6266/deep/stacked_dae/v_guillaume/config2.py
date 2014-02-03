'''
These are parameters used by nist_sda_retrieve.py. They'll end up as globals in there.

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

#Path of two pre-train done earlier
PATH_NIST = '/u/pannetis/IFT6266/ift6266/deep/stacked_dae/v_sylvain/NIST_big'
PATH_P07 = '/u/pannetis/IFT6266/ift6266/deep/stacked_dae/v_sylvain/P07_demo/'

# change "sandbox" when you're ready
JOBDB = 'postgres://ift6266h10@gershwin/ift6266h10_db/pannetis_SDA_retrieve'
EXPERIMENT_PATH = "ift6266.deep.stacked_dae.v_sylvain.nist_sda_retrieve.jobman_entrypoint"

##Pour lancer des travaux sur le cluster: (il faut etre ou se trouve les fichiers)
##python nist_sda_retrieve.py jobman_insert
##dbidispatch --condor --repeat_jobs=2 jobman sql 'postgres://ift6266h10@gershwin/ift6266h10_db/pannetis_finetuningSDA0' .  #C'est le path dans config.py

##Pour lancer sur GPU sur boltzmann (changer device=gpuX pour X le bon assigne)
##THEANO_FLAGS=floatX=float32,device=gpu2 python nist_sda_retrieve.py test_jobman_entrypoint


# reduce training set to that many examples
REDUCE_TRAIN_TO = None
# that's a max, it usually doesn't get to that point
MAX_FINETUNING_EPOCHS = 1000
# number of minibatches before taking means for valid error etc.
REDUCE_EVERY = 100
#Set the finetune dataset
FINETUNE_SET=0
#Set the pretrain dataset used. 0: NIST, 1:P07
PRETRAIN_CHOICE=0


if TEST_CONFIG:
    REDUCE_TRAIN_TO = 1000
    MAX_FINETUNING_EPOCHS = 2
    REDUCE_EVERY = 10


# This is to configure insertion of jobs on the cluster.
# Possible values the hyperparameters can take. These are then
# combined with produit_cartesien_jobs so we get a list of all
# possible combinations, each one resulting in a job inserted
# in the jobman DB.
JOB_VALS = {'pretraining_lr': [0.1],#, 0.001],#, 0.0001],
        'pretraining_epochs_per_layer': [10],
        'hidden_layers_sizes': [800],
        'corruption_levels': [0.2],
        'minibatch_size': [100],
        'max_finetuning_epochs':[MAX_FINETUNING_EPOCHS],
        'max_finetuning_epochs_P07':[1],
        'finetuning_lr':[0.01], #0.001 was very bad, so we leave it out
        'num_hidden_layers':[4],
        'finetune_set':[-1],
        'pretrain_choice':[0,1]
        }

# Just useful for tests... minimal number of epochs
# (This is used when running a single job, locally, when
# calling ./nist_sda.py test_jobman_entrypoint
DEFAULT_HP_NIST = {'finetuning_lr':0.1,
                       'pretraining_lr':0.01,
                       'pretraining_epochs_per_layer':15,
                       'max_finetuning_epochs':MAX_FINETUNING_EPOCHS,
                       #'max_finetuning_epochs':1,
                       'max_finetuning_epochs_P07':7,
                       'hidden_layers_sizes':1000,
                       'corruption_levels':0.2,
                       'minibatch_size':100,
                       #'reduce_train_to':2000,
		       'decrease_lr':1,
                       'num_hidden_layers':3,
                       'finetune_set':2,
                       'pretrain_choice':1}


