from dsetiter import DataIterator

class DataSet(object):
    def test(self, batchsize, bufsize=None): 
        r"""
        Returns an iterator over the test examples.

        Parameters
          batchsize (int) -- the size of the minibatches
          bufsize (int, optional) -- the size of the in-memory buffer,
                                     0 to disable.
        """
        return self._return_it(batchsize, bufsize, self._test)

    def train(self, batchsize, bufsize=None):
        r"""
        Returns an iterator over the training examples.

        Parameters
          batchsize (int) -- the size of the minibatches
          bufsize (int, optional) -- the size of the in-memory buffer,
                                     0 to disable.
        """
        return self._return_it(batchsize, bufsize, self._train)

    def valid(self, batchsize, bufsize=None):
        r"""
        Returns an iterator over the validation examples.

        Parameters
          batchsize (int) -- the size of the minibatches
          bufsize (int, optional) -- the size of the in-memory buffer,
                                     0 to disable.
        """
        return self._return_it(batchsize, bufsize, self._valid)

    def _return_it(batchsize, bufsize, data):
        r"""
        Must return an iterator over the specified dataset (`data`).

        Implement this in subclassses.
        """
        raise NotImplemented
