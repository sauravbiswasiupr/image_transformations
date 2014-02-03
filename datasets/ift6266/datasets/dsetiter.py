import numpy

class DummyFile(object):
    def __init__(self, size, shape=()):
        self.size = size
        self.shape = shape

    def read(self, num):
        if num > self.size:
            num = self.size
        self.size -= num
        return numpy.zeros((num,)+self.shape)

class DataIterator(object):
    
    def __init__(self, files, batchsize, bufsize=None):
        r""" 
        Makes an iterator which will read examples from `files`
        and return them in `batchsize` lots.

        Parameters: 
            files -- list of numpy readers
            batchsize -- (int) the size of returned batches
            bufsize -- (int, default=None) internal read buffer size.

        Tests:
            >>> d = DataIterator([DummyFile(930)], 10, 100)
            >>> d.batchsize
            10
            >>> d.bufsize
            100
            >>> d = DataIterator([DummyFile(1)], 10)
            >>> d.batchsize
            10
            >>> d.bufsize
            10000
            >>> d = DataIterator([DummyFile(1)], 99)
            >>> d.batchsize
            99
            >>> d.bufsize
            9999
            >>> d = DataIterator([DummyFile(1)], 10, 121)
            >>> d.batchsize
            10
            >>> d.bufsize
            120
            >>> d = DataIterator([DummyFile(1)], 10, 1)
            >>> d.batchsize
            10
            >>> d.bufsize
            10
            >>> d = DataIterator([DummyFile(1)], 2000)
            >>> d.batchsize
            2000
            >>> d.bufsize
            20000
            >>> d = DataIterator([DummyFile(1)], 2000, 31254)
            >>> d.batchsize
            2000
            >>> d.bufsize
            30000
            >>> d = DataIterator([DummyFile(1)], 2000, 10)
            >>> d.batchsize
            2000
            >>> d.bufsize
            2000
        """
        self.batchsize = batchsize
        if bufsize is None:
            self.bufsize = max(10*batchsize, 10000)
        else:
            self.bufsize = bufsize
        self.bufsize -= self.bufsize % self.batchsize
        if self.bufsize < self.batchsize:
            self.bufsize = self.batchsize
        self.files = iter(files)
        self.curfile = self.files.next()
        self.empty = False
        self._fill_buf()

    def _fill_buf(self):
        r"""
        Fill the internal buffer.

        Will fill across files in case the current one runs out.

        Test:
            >>> d = DataIterator([DummyFile(20, (3,2))], 10, 10)
            >>> d._fill_buf()
            >>> d.curpos
            0
            >>> len(d.buffer)
            10
            >>> d = DataIterator([DummyFile(11, (3,2)), DummyFile(9, (3,2))], 10, 10)
            >>> d._fill_buf()
            >>> len(d.buffer)
            10
            >>> d._fill_buf()
            Traceback (most recent call last):
              ...
            StopIteration
            >>> d = DataIterator([DummyFile(10, (3,2)), DummyFile(9, (3,2))], 10, 10)
            >>> d._fill_buf()
            >>> len(d.buffer)
            9
            >>> d._fill_buf()
            Traceback (most recent call last):
              ...
            StopIteration
            >>> d = DataIterator([DummyFile(20)], 10, 10)
            >>> d._fill_buf()
            >>> d.curpos
            0
            >>> len(d.buffer)
            10
            >>> d = DataIterator([DummyFile(11), DummyFile(9)], 10, 10)
            >>> d._fill_buf()
            >>> len(d.buffer)
            10
            >>> d._fill_buf()
            Traceback (most recent call last):
              ...
            StopIteration
            >>> d = DataIterator([DummyFile(10), DummyFile(9)], 10, 10)
            >>> d._fill_buf()
            >>> len(d.buffer)
            9
            >>> d._fill_buf()
            Traceback (most recent call last):
              ...
            StopIteration
        """
        self.buffer = None
        if self.empty:
            raise StopIteration
        buf = self.curfile.read(self.bufsize)
        
        while len(buf) < self.bufsize:
            try:
                self.curfile = self.files.next()
            except StopIteration:
                self.empty = True
                if len(buf) == 0:
                    raise
                break
            tmpbuf = self.curfile.read(self.bufsize - len(buf))
            buf = numpy.concatenate([buf, tmpbuf], axis=0)

        self.cursize = len(buf)
        self.buffer = buf
        self.curpos = 0

    def __next__(self):
        r"""
        Returns the next portion of the dataset.

        Test:
            >>> d = DataIterator([DummyFile(20)], 10, 20)
            >>> len(d.next())
            10
            >>> len(d.next())
            10
            >>> d.next()
            Traceback (most recent call last):
              ...
            StopIteration
            >>> d.next()
            Traceback (most recent call last):
              ...
            StopIteration
            >>> d = DataIterator([DummyFile(13)], 10, 50)
            >>> len(d.next())
            10
            >>> len(d.next())
            3
            >>> d.next()
            Traceback (most recent call last):
              ...
            StopIteration
        """
        if self.curpos >= self.cursize:
            self._fill_buf()
        res = self.buffer[self.curpos:self.curpos+self.batchsize]
        self.curpos += self.batchsize
        return res

    next = __next__

    def __iter__(self):
        return self
