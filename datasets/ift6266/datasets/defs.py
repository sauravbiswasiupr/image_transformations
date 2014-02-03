__all__ = ['nist_digits', 'nist_lower', 'nist_upper', 'nist_all', 'ocr', 
           'nist_P07', 'PNIST07', 'mnist']

from ftfile import FTDataSet
from gzpklfile import GzpklDataSet
import theano
import os

# if the environmental variables exist, get the path from them, 
# otherwise fall back on the default
NIST_PATH = os.getenv('NIST_PATH','/data/lisa/data/nist/by_class/')
DATA_PATH = os.getenv('DATA_PATH','/data/lisa/data/ift6266h10/')

nist_digits = lambda maxsize=None: FTDataSet(train_data = [os.path.join(NIST_PATH,'digits/digits_train_data.ft')],
                        train_lbl = [os.path.join(NIST_PATH,'digits/digits_train_labels.ft')],
                        test_data = [os.path.join(NIST_PATH,'digits/digits_test_data.ft')],
                        test_lbl = [os.path.join(NIST_PATH,'digits/digits_test_labels.ft')],
                        indtype=theano.config.floatX, inscale=255., maxsize=maxsize)
nist_lower = lambda maxsize=None: FTDataSet(train_data = [os.path.join(NIST_PATH,'lower/lower_train_data.ft')],
                        train_lbl = [os.path.join(NIST_PATH,'lower/lower_train_labels.ft')],
                        test_data = [os.path.join(NIST_PATH,'lower/lower_test_data.ft')],
                        test_lbl = [os.path.join(NIST_PATH,'lower/lower_test_labels.ft')],
                        indtype=theano.config.floatX, inscale=255., maxsize=maxsize)
nist_upper = lambda maxsize=None: FTDataSet(train_data = [os.path.join(NIST_PATH,'upper/upper_train_data.ft')],
                        train_lbl = [os.path.join(NIST_PATH,'upper/upper_train_labels.ft')],
                        test_data = [os.path.join(NIST_PATH,'upper/upper_test_data.ft')],
                        test_lbl = [os.path.join(NIST_PATH,'upper/upper_test_labels.ft')],
                        indtype=theano.config.floatX, inscale=255., maxsize=maxsize)

nist_all = lambda maxsize=None: FTDataSet(train_data = [os.path.join(DATA_PATH,'train_data.ft')],
                     train_lbl = [os.path.join(DATA_PATH,'train_labels.ft')],
                     test_data = [os.path.join(DATA_PATH,'test_data.ft')],
                     test_lbl = [os.path.join(DATA_PATH,'test_labels.ft')],
                     valid_data = [os.path.join(DATA_PATH,'valid_data.ft')],
                     valid_lbl = [os.path.join(DATA_PATH,'valid_labels.ft')],
                     indtype=theano.config.floatX, inscale=255., maxsize=maxsize)

ocr = lambda maxsize=None: FTDataSet(train_data = [os.path.join(DATA_PATH,'ocr_train_data.ft')],
                train_lbl = [os.path.join(DATA_PATH,'ocr_train_labels.ft')],
                test_data = [os.path.join(DATA_PATH,'ocr_test_data.ft')],
                test_lbl = [os.path.join(DATA_PATH,'ocr_test_labels.ft')],
                valid_data = [os.path.join(DATA_PATH,'ocr_valid_data.ft')],
                valid_lbl = [os.path.join(DATA_PATH,'ocr_valid_labels.ft')],
                indtype=theano.config.floatX, inscale=255., maxsize=maxsize)

#There is 2 more arguments here to can choose smaller datasets based on the file number.
#This is usefull to get different data for pre-training and finetuning
nist_P07 = lambda maxsize=None, min_file=0, max_file=100: FTDataSet(train_data = [os.path.join(DATA_PATH,'data/P07_train'+str(i)+'_data.ft') for i in range(min_file, max_file)],
                     train_lbl = [os.path.join(DATA_PATH,'data/P07_train'+str(i)+'_labels.ft') for i in range(min_file, max_file)],
                     test_data = [os.path.join(DATA_PATH,'data/P07_test_data.ft')],
                     test_lbl = [os.path.join(DATA_PATH,'data/P07_test_labels.ft')],
                     valid_data = [os.path.join(DATA_PATH,'data/P07_valid_data.ft')],
                     valid_lbl = [os.path.join(DATA_PATH,'data/P07_valid_labels.ft')],
                     indtype=theano.config.floatX, inscale=255., maxsize=maxsize)
		     
#Added PNIST07
PNIST07 = lambda maxsize=None, min_file=0, max_file=100: FTDataSet(train_data = [os.path.join(DATA_PATH,'data/PNIST07_train'+str(i)+'_data.ft') for i in range(min_file, max_file)],
                     train_lbl = [os.path.join(DATA_PATH,'data/PNIST07_train'+str(i)+'_labels.ft') for i in range(min_file, max_file)],
                     test_data = [os.path.join(DATA_PATH,'data/PNIST07_test_data.ft')],
                     test_lbl = [os.path.join(DATA_PATH,'data/PNIST07_test_labels.ft')],
                     valid_data = [os.path.join(DATA_PATH,'data/PNIST07_valid_data.ft')],
                     valid_lbl = [os.path.join(DATA_PATH,'data/PNIST07_valid_labels.ft')],
                     indtype=theano.config.floatX, inscale=255., maxsize=maxsize)

mnist = lambda maxsize=None: GzpklDataSet(os.path.join(DATA_PATH,'mnist.pkl.gz'),
                                          maxsize=maxsize)
