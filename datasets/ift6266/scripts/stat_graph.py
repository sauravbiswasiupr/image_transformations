import matplotlib
matplotlib.use('Agg')

from pylab import *
from scipy import stats
import numpy
from ift6266 import datasets

nistp_valid = stats.itemfreq(datasets.PNIST07().valid(10000000).next()[1])
nistp_valid[:,1] /= sum(nistp_valid[:,1])
nist_valid = stats.itemfreq(datasets.nist_all().valid(10000000).next()[1])
nist_valid[:,1] /= sum(nist_valid[:,1])
nist_test = stats.itemfreq(datasets.nist_all().test(10000000).next()[1])
nist_test[:,1] /= sum(nist_test[:,1])
nist_train = stats.itemfreq(datasets.nist_all().train(100000000).next()[1])
nist_train[:,1] /= sum(nist_train[:,1])

xloc = numpy.arange(62)+0.5

labels = map(str, range(10)) + map(chr, range(65,91)) + map(chr, range(97,123))

def makegraph(data, fname, labels=labels, xloc=xloc, width=0.5):
    figure(figsize=(8,6))
#    clf()
    bar(xloc, data, width=width)
    xticks([])
    for x, l in zip(xloc, labels):
        text(x+width/2, -0.004, l, horizontalalignment='center', verticalalignment='baseline')
#    xticks(xloc+width/2, labels, verticalalignment='bottom')
    xlim(0, xloc[-1]+width*2)
    ylim(0, 0.1)

    savefig(fname)


makegraph(nistp_valid[:,1], 'nistpvalidstats.png')
makegraph(nist_valid[:,1], 'nistvalidstats.png')
makegraph(nist_test[:,1], 'nistteststats.png')
makegraph(nist_train[:,1], 'nisttrainstats.png')
