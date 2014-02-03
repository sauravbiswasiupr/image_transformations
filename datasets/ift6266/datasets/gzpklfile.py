import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle

from dataset import DataSet
from dsetiter import DataIterator
from itertools import izip

class ArrayFile(object):
    def __init__(self, ary):
        self.ary = ary
        self.pos = 0

    def read(self, num):
        res = self.ary[self.pos:self.pos+num]
        self.pos += num
        return res

class GzpklDataSet(DataSet):
    def __init__(self, fname, maxsize):
        self._fname = fname
        self.maxsize = maxsize
        self._train = 0
        self._valid = 1
        self._test = 2

    def _load(self):
        f = gzip.open(self._fname, 'rb')
        try:
            self.datas = pickle.load(f)
        finally:
            f.close()

    def _return_it(self, batchsz, bufsz, id):
        if not hasattr(self, 'datas'):
            self._load()
        return izip(DataIterator([ArrayFile(self.datas[id][0][:self.maxsize])], batchsz, bufsz),
                    DataIterator([ArrayFile(self.datas[id][1][:self.maxsize])], batchsz, bufsz))
