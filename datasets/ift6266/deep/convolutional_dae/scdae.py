from pynnet import *

import numpy
import theano
import theano.tensor as T

from itertools import izip
from ift6266.utils.seriestables import *

class cdae(LayerStack):
    def __init__(self, filter_size, num_filt, num_in, subsampling, corruption,
                 dtype):
        LayerStack.__init__(self, [ConvAutoencoder(filter_size=filter_size, 
                                                   num_filt=num_filt,
                                                   num_in=num_in,
                                                   noise=corruption,
                                                   err=errors.cross_entropy,
                                                   nlin=nlins.sigmoid,
                                                   dtype=dtype),
                                   MaxPoolLayer(subsampling)])

    def build(self, input, input_shape=None):
        LayerStack.build(self, input, input_shape)
        self.cost = self.layers[0].cost
        self.pre_params = self.layers[0].pre_params

def scdae(filter_sizes, num_filts, subsamplings, corruptions, dtype):
    layers = []
    old_nfilt = 1
    for fsize, nfilt, subs, corr in izip(filter_sizes, num_filts,
                                         subsamplings, corruptions):
        layers.append(cdae(fsize, nfilt, old_nfilt, subs, corr, dtype))
        old_nfilt = nfilt
    return LayerStack(layers, name='scdae')

def mlp(layer_sizes, dtype):
    layers = []
    old_size = layer_sizes[0]
    for size in layer_sizes[1:]:
        layers.append(SimpleLayer(old_size, size, activation=nlins.tanh,
                                  dtype=dtype))
        old_size = size
    return LayerStack(layers, name='mlp')

def scdae_net(in_size, filter_sizes, num_filts, subsamplings,
              corruptions, layer_sizes, out_size, dtype):
    rl1 = ReshapeLayer((None,)+in_size)
    ls = scdae(filter_sizes, num_filts, subsamplings, 
               corruptions, dtype)
    x = T.ftensor4()
    ls.build(x, input_shape=(1,)+in_size)
    outs = numpy.prod(ls.output_shape)
    rl2 = ReshapeLayer((None, outs))
    layer_sizes = [outs]+layer_sizes
    ls2 = mlp(layer_sizes, dtype)
    lrl = SimpleLayer(layer_sizes[-1], out_size, activation=nlins.softmax, 
                      name='output')
    return NNet([rl1, ls, rl2, ls2, lrl], error=errors.nll)

def build_funcs(batch_size, img_size, filter_sizes, num_filters, subs,
                noise, mlp_sizes, out_size, dtype, pretrain_lr, train_lr):
    
    n = scdae_net((1,)+img_size, filter_sizes, num_filters, subs,
                  noise, mlp_sizes, out_size, dtype)

    n.save('start.net')

    x = T.fmatrix('x')
    y = T.ivector('y')
    
    def pretrainfunc(net, alpha):
        up = trainers.get_updates(net.pre_params, net.cost, alpha)
        return theano.function([x], net.cost, updates=up)

    def trainfunc(net, alpha):
        up = trainers.get_updates(net.params, net.cost, alpha)
        return theano.function([x, y], net.cost, updates=up)

    n.build(x, y, input_shape=(batch_size, numpy.prod(img_size)))
    pretrain_funcs_opt = [pretrainfunc(l, pretrain_lr) for l in n.layers[1].layers]
    trainf_opt = trainfunc(n, train_lr)
    evalf_opt = theano.function([x, y], errors.class_error(n.output, y))
    
    n.build(x, y)
    pretrain_funcs_reg = [pretrainfunc(l, 0.01) for l in n.layers[1].layers]
    trainf_reg = trainfunc(n, 0.1)
    evalf_reg = theano.function([x, y], errors.class_error(n.output, y))

    def select_f(f1, f2, bsize):
        def f(x):
            if x.shape[0] == bsize:
                return f1(x)
            else:
                return f2(x)
        return f
    
    pretrain_funcs = [select_f(p_opt, p_reg, batch_size) for p_opt, p_reg in zip(pretrain_funcs_opt, pretrain_funcs_reg)]
    
    def select_f2(f1, f2, bsize):
        def f(x, y):
            if x.shape[0] == bsize:
                return f1(x, y)
            else:
                return f2(x, y)
        return f

    trainf = select_f2(trainf_opt, trainf_reg, batch_size)
    evalf = select_f2(evalf_opt, evalf_reg, batch_size)
    return pretrain_funcs, trainf, evalf, n

def do_pretrain(pretrain_funcs, pretrain_epochs, serie):
    for layer, f in enumerate(pretrain_funcs):
        for epoch in xrange(pretrain_epochs):
            serie.append((layer, epoch), f())

def massage_funcs(pretrain_it, train_it, dset, batch_size, pretrain_funcs,
                  trainf, evalf):
    def pretrain_f(f):
        def res():
            for x, y in pretrain_it:
                yield f(x)
        it = res()
        return lambda: it.next()

    pretrain_fs = map(pretrain_f, pretrain_funcs)

    def train_f(f):
        def dset_it():
            for x, y in train_it:
                yield f(x, y)
        it = dset_it()
        return lambda: it.next()
    
    train = train_f(trainf)
    
    def eval_f(f, dsetf):
        def res():
            c = 0
            i = 0
            for x, y in dsetf(batch_size):
                i += x.shape[0]
                c += f(x, y)*x.shape[0]
            return c/i
        return res
    
    test = eval_f(evalf, dset.test)
    valid = eval_f(evalf, dset.valid)

    return pretrain_fs, train, valid, test

def repeat_itf(itf, *args, **kwargs):
    while True:
        for e in itf(*args, **kwargs):
            yield e

def create_series():
    import tables

    series = {}
    h5f = tables.openFile('series.h5', 'w')
    class PrintWrap(object):
        def __init__(self, series):
            self.series = series

        def append(self, idx, value):
            print idx, value
            self.series.append(idx, value)

    series['recons_error'] = AccumulatorSeriesWrapper(
        base_series=PrintWrap(ErrorSeries(error_name='reconstruction_error',
                                          table_name='reconstruction_error',
                                          hdf5_file=h5f,
                                          index_names=('layer', 'epoch'),
                                          title="Reconstruction error (mse)")),
        reduce_every=100)
        
    series['train_error'] = AccumulatorSeriesWrapper(
        base_series=ErrorSeries(error_name='training_error',
                                table_name='training_error',
                                hdf5_file=h5f,
                                index_names=('iter',),
                                title='Training error (nll)'),
        reduce_every=100)
    
    series['valid_error'] = ErrorSeries(error_name='valid_error',
                                        table_name='valid_error',
                                        hdf5_file=h5f,
                                        index_names=('iter',),
                                        title='Validation error (class)')
    
    series['test_error'] = ErrorSeries(error_name='test_error',
                                       table_name='test_error',
                                       hdf5_file=h5f,
                                       index_names=('iter',),
                                       title='Test error (class)')
    
    return series

class PrintSeries(object):
    def append(self, idx, v):
        print idx, v

if __name__ == '__main__':
    from ift6266 import datasets
    from sgd_opt import sgd_opt
    import sys, time
    
    batch_size = 100
    dset = datasets.nist_digits(1000)

    pretrain_funcs, trainf, evalf, net = build_funcs(
        img_size = (32, 32),
        batch_size=batch_size, filter_sizes=[(5,5), (3,3)],
        num_filters=[20, 4], subs=[(2,2), (2,2)], noise=[0.2, 0.2],
        mlp_sizes=[500], out_size=10, dtype=numpy.float32,
        pretrain_lr=0.001, train_lr=0.1)
    
    t_it = repeat_itf(dset.train, batch_size)
    pretrain_fs, train, valid, test = massage_funcs(
        t_it, t_it, dset, batch_size,
        pretrain_funcs, trainf, evalf)

    print "pretraining ...",
    sys.stdout.flush()
    start = time.time()
    do_pretrain(pretrain_fs, 1000, PrintSeries())
    end = time.time()
    print "done (in", end-start, "s)"
    
    sgd_opt(train, valid, test, training_epochs=10000, patience=1000,
            patience_increase=2., improvement_threshold=0.995,
            validation_frequency=250)

