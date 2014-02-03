#!/usr/bin/python
# coding: utf-8

import sys
import tempfile
import os.path
import os

import numpy

from series import BaseSeries, AccumulatorSeries, SeriesContainer, BasicStatsSeries, SeriesMultiplexer, SeriesList, ParamsArrayStats


BASEDIR = tempfile.mkdtemp()

def tempname():
    file = tempfile.NamedTemporaryFile(dir=BASEDIR)
    filepath = file.name
    return os.path.split(filepath)

def tempdir():
    wholepath = os.path.dirname(tempfile.mkdtemp(dir=BASEDIR))
    # split again, interpreting the last directory as a filename
    return os.path.split(wholepath)

def tempseries(type='f', flush_every=1):
    dir, filename = tempname()

    s = BaseSeries(name=filename, directory=dir, type=type, flush_every=flush_every)

    return s

def test_Series_storeload():
    s = tempseries()

    s.append(12.0)
    s.append_list([13.0,14.0,15.0])

    s2 = BaseSeries(name=s.name, directory=s.directory, flush_every=15)
    # also test if elements stored before load_from_file (and before a flush)
    # are deleted (or array is restarted from scratch... both work)
    s2.append(10.0)
    s2.append_list([30.0,40.0])
    s2.load_from_file()

    assert s2.tolist() == [12.0,13.0,14.0,15.0]


def test_AccumulatorSeries_mean():
    dir, filename = tempname()

    s = AccumulatorSeries(reduce_every=15, mean=True, name=filename, directory=dir)

    for i in range(50):
        s.append(i)

    assert s.tolist() == [7.0,22.0,37.0]

def test_BasicStatsSeries_commoncase():
    a1 = numpy.arange(25).reshape((5,5))
    a2 = numpy.arange(40).reshape((8,5))
    
    parent_dir, dir = tempdir()

    bss = BasicStatsSeries(parent_directory=parent_dir, name=dir)

    bss.append(a1)
    bss.append(a2)

    assert bss.means.tolist() == [12.0, 19.5]
    assert bss.mins.tolist() == [0.0, 0.0]
    assert bss.maxes.tolist() == [24.0, 39.0]
    assert (bss.stds.tolist()[0] - 7.211102) < 1e-3
    assert (bss.stds.tolist()[1] - 11.54339) < 1e-3

    # try to reload

    bss2 = BasicStatsSeries(parent_directory=parent_dir, name=dir)
    bss2.load_from_directory()

    assert bss2.means.tolist() == [12.0, 19.5]
    assert bss2.mins.tolist() == [0.0, 0.0]
    assert bss2.maxes.tolist() == [24.0, 39.0]
    assert (bss2.stds.tolist()[0] - 7.211102) < 1e-3
    assert (bss2.stds.tolist()[1] - 11.54339) < 1e-3

def test_BasicStatsSeries_reload():
    a1 = numpy.arange(25).reshape((5,5))
    a2 = numpy.arange(40).reshape((8,5))
    
    parent_dir, dir = tempdir()

    bss = BasicStatsSeries(parent_directory=parent_dir, name=dir)

    bss.append(a1)
    bss.append(a2)

    # try to reload

    bss2 = BasicStatsSeries(parent_directory=parent_dir, name=dir)
    bss2.load_from_directory()

    assert bss2.means.tolist() == [12.0, 19.5]
    assert bss2.mins.tolist() == [0.0, 0.0]
    assert bss2.maxes.tolist() == [24.0, 39.0]
    assert (bss2.stds.tolist()[0] - 7.211102) < 1e-3
    assert (bss2.stds.tolist()[1] - 11.54339) < 1e-3


def test_BasicStatsSeries_withaccumulator():
    a1 = numpy.arange(25).reshape((5,5))
    a2 = numpy.arange(40).reshape((8,5))
    a3 = numpy.arange(20).reshape((4,5))
    a4 = numpy.arange(48).reshape((6,8))
    
    parent_dir, dir = tempdir()

    sc = AccumulatorSeries.series_constructor(reduce_every=2, mean=False)

    bss = BasicStatsSeries(parent_directory=parent_dir, name=dir, series_constructor=sc)

    bss.append(a1)
    bss.append(a2)
    bss.append(a3)
    bss.append(a4)

    assert bss.means.tolist() == [31.5, 33.0]

def test_SeriesList_withbasicstats():
    dir = tempfile.mkdtemp(dir=BASEDIR)

    bscstr = BasicStatsSeries.series_constructor()

    slist = SeriesList(num_elements=5, name="foo", directory=dir, series_constructor=bscstr)

    for i in range(10): # 10 elements in each list
        curlist = []
        for j in range(5): # 5 = num_elements, ie. number of list to append to
            dist = numpy.arange(i*j, i*j+10)
            curlist.append(dist)
        slist.append(curlist)

    slist2 = SeriesList(num_elements=5, name="foo", directory=dir, series_constructor=bscstr)

    slist2.load_from_files()

    l1 = slist2._subseries[0].means.tolist()
    l2 = slist2._subseries[4].means.tolist()

    print l1
    print l2

    assert l1 == [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
    assert l2 == [4.5, 8.5, 12.5, 16.5, 20.5, 24.5, 28.5, 32.5, 36.5, 40.5]

# same test as above, just with the shortcut
def test_ParamsArrayStats_reload():
    dir = tempfile.mkdtemp(dir=BASEDIR)

    slist = ParamsArrayStats(5, name="foo", directory=dir)

    for i in range(10): # 10 elements in each list
        curlist = []
        for j in range(5): # 5 = num_elements, ie. number of list to append to
            dist = numpy.arange(i*j, i*j+10)
            curlist.append(dist)
        slist.append(curlist)

    slist2 = ParamsArrayStats(5, name="foo", directory=dir)

    slist2.load_from_files()

    l1 = slist2._subseries[0].means.tolist()
    l2 = slist2._subseries[4].means.tolist()

    print l1
    print l2

    assert l1 == [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
    assert l2 == [4.5, 8.5, 12.5, 16.5, 20.5, 24.5, 28.5, 32.5, 36.5, 40.5]


def manual_BasicStatsSeries_graph():
    parent_dir, dir = tempdir()

    bss = BasicStatsSeries(parent_directory=parent_dir, name=dir)

    for i in range(50):
        bss.append(1.0/numpy.arange(i*5, i*5+5))

    bss.graph()

#if __name__ == '__main__':
#    import pylab
#    manual_BasicStatsSeries_graph()
#    pylab.show()

