from itertools import izip
import os

import numpy
from pylearn.io.filetensor import _read_header, _prod

from dataset import DataSet
from dsetiter import DataIterator


class FTFile(object):
    def __init__(self, fname, scale=1, dtype=None):
        r"""
        Tests:
            >>> f = FTFile('/data/lisa/data/nist/by_class/digits/digits_test_labels.ft')
        """
        if os.path.exists(fname):
            self.file = open(fname, 'rb')
            self.magic_t, self.elsize, _, self.dim, _ = _read_header(self.file, False)
            self.gz=False
        else:
            import gzip
            self.file = gzip.open(fname+'.gz','rb')
            self.magic_t, self.elsize, _, self.dim, _ = _read_header(self.file, False, True)
            self.gz=True

        self.size = self.dim[0]
        self.scale = scale
        self.dtype = dtype

    def skip(self, num):
        r"""
        Skips `num` items in the file.

        If `num` is negative, skips size-num.

        Tests:
            >>> f = FTFile('/data/lisa/data/nist/by_class/digits/digits_test_labels.ft')
            >>> f.size
            58646
            >>> f.elsize
            4
            >>> f.file.tell()
            20
            >>> f.skip(1000)
            >>> f.file.tell()
            4020
            >>> f.size
            57646
            >>> f = FTFile('/data/lisa/data/nist/by_class/digits/digits_test_labels.ft')
            >>> f.size
            58646
            >>> f.file.tell()
            20
            >>> f.skip(-1000)
            >>> f.file.tell()
            230604
            >>> f.size
            1000
        """
        if num < 0:
            num += self.size
        if num < 0:
            raise ValueError('Skipping past the start of the file')
        if num >= self.size:
            self.size = 0
        else:
            self.size -= num
            f_start = self.file.tell()
            self.file.seek(f_start + (self.elsize * _prod(self.dim[1:]) * num))
    
    def read(self, num):
        r"""
        Reads `num` elements from the file and return the result as a
        numpy matrix.  Last read is truncated.

        Tests:
            >>> f = FTFile('/data/lisa/data/nist/by_class/digits/digits_test_labels.ft')
            >>> f.read(1)
            array([6], dtype=int32)
            >>> f.read(10)
            array([7, 4, 7, 5, 6, 4, 8, 0, 9, 6], dtype=int32)
            >>> f.skip(58630)
            >>> f.read(10)
            array([9, 2, 4, 2, 8], dtype=int32)
            >>> f.read(10)
            array([], dtype=int32)
            >>> f = FTFile('/data/lisa/data/nist/by_class/digits/digits_test_data.ft')
            >>> f.read(1)
            array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
        """
        if num > self.size:
            num = self.size
        self.dim[0] = num
        self.size -= num
        if self.gz:
            d = self.file.read(_prod(self.dim)*self.elsize)
            res = numpy.fromstring(d, dtype=self.magic_t, count=_prod(self.dim)).reshape(self.dim)
        else:
            res = numpy.fromfile(self.file, dtype=self.magic_t, count=_prod(self.dim)).reshape(self.dim)
        if self.dtype is not None:
            res = res.astype(self.dtype)
        if self.scale != 1:
            res /= self.scale
        return res

class FTSource(object):
    def __init__(self, file, skip=0, size=None, maxsize=None, 
                 dtype=None, scale=1):
        r"""
        Create a data source from a possible subset of a .ft file.

        Parameters:
            `file` -- (string) the filename
            `skip` -- (int, optional) amount of examples to skip from
                      the start of the file.  If negative, skips
                      filesize - skip.
            `size` -- (int, optional) truncates number of examples
                      read (after skipping).  If negative truncates to
                      filesize - size (also after skipping).
            `maxsize` -- (int, optional) the maximum size of the file
            `dtype` -- (dtype, optional) convert the data to this
                       dtype after reading.
            `scale` -- (number, optional) scale (that is divide) the
                       data by this number (after dtype conversion, if
                       any).

        Tests:
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft')
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft', size=1000)
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft', skip=10)
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft', skip=100, size=120)
        """
        self.file = file
        self.skip = skip
        self.size = size
        self.dtype = dtype
        self.scale = scale
        self.maxsize = maxsize
    
    def open(self):
        r"""
        Returns an FTFile that corresponds to this dataset.
        
        Tests:
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft')
        >>> f = s.open()
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft', size=1)
        >>> len(s.open().read(2))
        1
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft', skip=57646)
        >>> s.open().size
        1000
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft', skip=57646, size=1)
        >>> s.open().size
        1
        >>> s = FTSource('/data/lisa/data/nist/by_class/digits/digits_test_data.ft', size=-10)
        >>> s.open().size
        58636
        """
        f = FTFile(self.file, scale=self.scale, dtype=self.dtype)
        if self.skip != 0:
            f.skip(self.skip)
        if self.size is not None and self.size < f.size:
            if self.size < 0:
                f.size += self.size
                if f.size < 0:
                    f.size = 0
            else:
                f.size = self.size
        if self.maxsize is not None and f.size > self.maxsize:
            f.size = self.maxsize
        return f

class FTData(object):
    r"""
    This is a list of FTSources.
    """
    def __init__(self, datafiles, labelfiles, skip=0, size=None, maxsize=None,
                 inscale=1, indtype=None, outscale=1, outdtype=None):
        if maxsize is not None:
            maxsize /= len(datafiles)
        self.inputs = [FTSource(f, skip, size, maxsize, scale=inscale, dtype=indtype)
                       for f in  datafiles]
        self.outputs = [FTSource(f, skip, size, maxsize, scale=outscale, dtype=outdtype)
                        for f in labelfiles]

    def open_inputs(self):
        return [f.open() for f in self.inputs]

    def open_outputs(self):
        return [f.open() for f in self.outputs]
    

class FTDataSet(DataSet):
    def __init__(self, train_data, train_lbl, test_data, test_lbl, 
                 valid_data=None, valid_lbl=None, indtype=None, outdtype=None,
                 inscale=1, outscale=1, maxsize=None):
        r"""
        Defines a DataSet from a bunch of files.
        
        Parameters:
           `train_data` -- list of train data files
           `train_label` -- list of train label files (same length as `train_data`)
           `test_data`, `test_labels` -- same thing as train, but for
                                         test.  The number of files
                                         can differ from train.
           `valid_data`, `valid_labels` -- same thing again for validation.
                                           (optional)
           `indtype`, `outdtype`,  -- see FTSource.__init__()
           `inscale`, `outscale`      (optional)
           `maxsize` -- maximum size of the set returned
                                                             

        If `valid_data` and `valid_labels` are not supplied then a sample
        approximately equal in size to the test set is taken from the train 
        set.
        """
        if valid_data is None:
            total_valid_size = sum(FTFile(td).size for td in test_data)
            if maxsize is not None:
                total_valid_size = min(total_valid_size, maxsize) 
            valid_size = total_valid_size/len(train_data)
            self._train = FTData(train_data, train_lbl, size=-valid_size,
                                 inscale=inscale, outscale=outscale,
                                 indtype=indtype, outdtype=outdtype,
                                 maxsize=maxsize)
            self._valid = FTData(train_data, train_lbl, skip=-valid_size,
                                 inscale=inscale, outscale=outscale,
                                 indtype=indtype, outdtype=outdtype,
                                 maxsize=maxsize)
        else:
            self._train = FTData(train_data, train_lbl, maxsize=maxsize,
                                 inscale=inscale, outscale=outscale, 
                                 indtype=indtype, outdtype=outdtype)
            self._valid = FTData(valid_data, valid_lbl, maxsize=maxsize,
                                 inscale=inscale, outscale=outscale,
                                 indtype=indtype, outdtype=outdtype)
        self._test = FTData(test_data, test_lbl, maxsize=maxsize,
                            inscale=inscale, outscale=outscale,
                            indtype=indtype, outdtype=outdtype)

    def _return_it(self, batchsize, bufsize, ftdata):
        return izip(DataIterator(ftdata.open_inputs(), batchsize, bufsize),
                    DataIterator(ftdata.open_outputs(), batchsize, bufsize))
