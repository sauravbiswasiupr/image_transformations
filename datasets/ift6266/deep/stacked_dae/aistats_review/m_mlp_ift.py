import pdb,bricks.costs,datetime,os,theano,sys
from bricks.experiments import *
from bricks.networks import *
from bricks import *
from datasets import *
from bricks.optimizer import *
from monitor.exp_monitoring import *

#from monitor.series import *
import numpy
#import jobman,jobman.sql,pylearn.version
#from jobman import DD
#from utils.JobmanHandling import *

class MnistTSdaeExperiment(ExperimentObject):
    # Todo : Write down the interface

    def _init_dataset(self):
        self.dataset_list = [ PNIST07(), nist_all() ]
        self.dataset = self.dataset_list[0]


    def _init_outputs(self):
        self.ds_output = { 'Pnist_Train' : self.dataset_list[0].train,
                           'Pnist_Valid' : self.dataset_list[0].valid,
                           'Pnist_Test' : self.dataset_list[0].test,
                           'nist_Train' : self.dataset_list[1].train,
                           'nist_Valid' : self.dataset_list[1].valid,
                           'nist_Test' : self.dataset_list[1].test}

        self.outputs = { 'CC' : costs.classification_error(self.network.layers[-1][0].out_dict['argmax_softmax_output'],self.network.in_dict['pred']) }
                         #'L1' : costs.L1(self.network.layers[0][0].out_dict['sigmoid_output']) }
                         #'LL' : costs.negative_ll(self.network.layers[-1][0].out_dict['softmax_output'],self.network.in_dict['pred']) }



    def _init_network(self):
        """ Choose wich network to initialize """
        #x,y = self.dataset.train(1).next()
        n_i = 1024
        n_o = 62
        numpy.random.seed(self.hp['seed'])
        self.network = MLPNetwork(n_i,n_o,size=self.hp['size'])
        default.load_pickled_network(self.network,'best_params/1/')

    def _init_costs_params(self):
        #finetuning
        self.costs  = [ [costs.negative_ll(self.network.layers[-1][0].out_dict['softmax_output'],self.network.in_dict['pred'])] ]
        self.params = [ [self.network.get_all_params(),self.network.get_all_params()] ]


    def _init_monitor(self):
        self.monitor = monitor(self.outputs,self.ds_output,self.network,self.sub_paths,save_criterion='Pnist_Valid')

    def startexp(self):
        print self.info()
        for j,optimizer in enumerate(self.optimizers):
            print 'Optim', '#'+str(j+1)
            sys.stdout.flush()
            for i in range(self.hp['ft_ep']):
                optimizer.tune(self.dataset.train,self.hp['bs'])
                print repr(i).rjust(3),self.monitor.get_str_output()
                sys.stdout.flush()


    def run(self):
        self.startexp()
        self.monitor.dump()
        return True

def jobman_entrypoint(state, channel):
    import jobman,jobman.sql,pylearn.version
    from jobman import DD
    from utils.JobmanHandling import JobHandling,jobman_insert,cartesian_product_jobs
    exp = MnistTSdaeExperiment(state,channel)
    return exp.jobhandler.start(state,channel)

def standalone(state):
    exp = MnistTSdaeExperiment(state)
    exp.run()


if __name__ == '__main__':
    HP = { 'lr':[ [ .1] ],
           'ft_ep':[100],
           'bs':[100],
           'size':[ [300],[4000],[5000],[6000],[7000] ],
           'seed':[0]}

    job_db_path = 'postgres://mullerx:b9f6ed1ee4@gershwin/mullerx_db/m_mlp_ift'
    exp_path = "m_mlp_ift.jobman_entrypoint"

    args = sys.argv[1:]

    if len(args) > 0 and args[0] == 'jobman_insert':
        jobman_insert(HP,job_db_path,exp_path)

    elif len(args) > 0 and args[0] == 'jobman_test':
        chanmock = DD({'COMPLETE':0,'save':(lambda:None)})
        dd_hp = cartesian_product_jobs(HP)
        print dd_hp[0]
        jobman_entrypoint(dd_hp[0], chanmock)

    elif len(args) > 0 and args[0] == 'standalone':
        hp = { 'lr':[ .1],
           'ft_ep':100,  
           'bs':100,
           'size':[ 3000 ],
           'seed':0}
        standalone(hp)
        
        
    else:
        print "Bad arguments"


#jobman sqlview  postgres://mullerx:b9f6ed1ee4@gershwin/mullerx_db/m_mlp_ift m_mlp_ift_view
#psql -h gershwin -U mullerx -d mullerx_db
#b9f6ed1ee4

#jobdispatch --condor  --env=THEANO_FLAGS=floatX=float32 --repeat_jobs=5 jobman sql -n0 'postgres://mullerx:b9f6ed1ee4@gershwin/mullerx_db/m_mlp_ift' .
