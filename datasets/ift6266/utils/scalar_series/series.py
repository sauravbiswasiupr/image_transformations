#!/usr/bin/python
# coding: utf-8

from __future__ import with_statement

import sys
import os
import os.path
import array

# for BasicStatsSeries
import numpy

# To access .value if necessary
import theano.tensor.sharedvar

'''
* TODO: add xy series
* TODO: add graph() for base and accumulator
* TODO: flush_every for BaseStatsSeries
* TODO: warn when Mux append() is called with a nonexisting name
* SeriesContainers are also series, albeit with more complex elements appended
* Each series has a "name" which corresponds in some way to the directory or file in which it's saved
'''

# Simple class to append numbers and flush them to a file once in a while
class BaseSeries():
    # for types, see http://docs.python.org/library/array.html
    def __init__(self, name, directory, type='f', flush_every=1):
        self.type = type
        self.flush_every = flush_every

        if not name or not directory:
            raise Exception("name and directory must be provided (strings)")

        self.directory = directory
        self.name = name

        if name and directory:
            self.filepath = os.path.join(directory, name)

        self._array = array.array(type)
        # stores the length not stored in file, waiting to be flushed
        self._buffered = 0

    def append(self, newitem):
        self._array.append(newitem)

        self._buffered += 1
        if self._buffered >= self.flush_every:
            self.flush()

    def append_list(self, items):
        self._array.fromlist(items)
        self._buffered += len(items)
        if self._buffered >= self.flush_every:
            self.flush()

    def flush(self):
        if self._buffered == 0:
            return
        with open(self.filepath, "wb") as f:
            s = self._array[-self._buffered:].tostring()
            f.write(s)

    def tolist(self):
        return self._array.tolist()

    def load_from_file(self):
        if not self.filepath:
            raise Exception("No name/directory provided")

        self._array = array.array(self.type)
        self._buffered = 0

        statinfo = os.stat(self.filepath)
        size = statinfo.st_size
        num_items = size / self._array.itemsize

        with open(self.filepath, "rb") as f:
            self._array.fromfile(f, num_items)

class AccumulatorSeries(BaseSeries):
    '''
    reduce_every: group (sum or mean) the last "reduce_every" items whenever we have enough
                and create a new item added to the real, saved array
                (if elements remain at the end, less then "reduce_every", they'll be discarded on program close)
    flush_every: this is for items of the real, saved array, not in terms of number of calls to "append"
    '''
    def __init__(self, reduce_every,
                    name, directory, flush_every=1,
                    mean=False):
        BaseSeries.__init__(self, name=name, directory=directory, type='f', flush_every=flush_every)
        self.reduce_every = reduce_every
        self._accumulator = 0.0
        self._num_accumulated = 0
        self.use_mean = mean

    @classmethod
    def series_constructor(cls, reduce_every, mean=False):
        def cstr(name, directory, flush_every=1):
            return cls(reduce_every=reduce_every, mean=mean, name=name, directory=directory, flush_every=flush_every)
        return cstr

    def append(self, item):
        self._accumulator += item
        self._num_accumulated += 1
        if self._num_accumulated >= self.reduce_every:
            n = self._accumulator
            if self.use_mean:
                n = n / self.reduce_every
            BaseSeries.append(self, n)

            self._num_accumulated = 0
            self._accumulator = 0.0

    def append_list(self, items):
        for i in items:
            self.append(i)

class SeriesContainer():
    def __init__(self, parent_directory, name,
                    series_constructor=BaseSeries):
        self.parent_directory = parent_directory
        self.name = name

        if not parent_directory or not name:
            raise Exception("parent_directory and name must be provided (strings)")

        self.directory_path = os.path.join(parent_directory, name)

        self.series_constructor = series_constructor

        # attempt to create directory for series
        if not os.path.isdir(self.directory_path):
            os.mkdir(self.directory_path)

    def graph(self):
        pass

class BasicStatsSeries(SeriesContainer):
    def __init__(self, parent_directory, name, series_constructor=BaseSeries,
            mean=True, minmax=True, std=True):
        SeriesContainer.__init__(self, parent_directory=parent_directory, name=name, series_constructor=series_constructor)

        self.save_mean = mean
        self.save_minmax = minmax
        self.save_std = std

        self.create_series()

    @classmethod
    def series_constructor(cls, mean=True, minmax=True, std=True):
        def cstr(name, directory, flush_every=1):
            return cls(name=name, parent_directory=directory,
                        mean=mean, minmax=minmax, std=std)
        return cstr


    def create_series(self):
        if self.save_mean:
            self.means = self.series_constructor(name="mean", directory=self.directory_path)

        if self.save_minmax:
            self.mins = self.series_constructor(name="min", directory=self.directory_path)
            self.maxes = self.series_constructor(name="max", directory=self.directory_path)

        if self.save_std:
            self.stds = self.series_constructor(name="std", directory=self.directory_path)

    def append(self, array):
        # TODO: shouldn't this be the job of the caller? (at least ParamsArraySeries)
        if isinstance(array, theano.tensor.sharedvar.TensorSharedVariable):
            array = array.value

        if self.save_mean:
            n = numpy.mean(array)
            self.means.append(n)
        if self.save_minmax:
            n = numpy.min(array)
            self.mins.append(n)
            n = numpy.max(array)
            self.maxes.append(n)
        if self.save_std:
            n = numpy.std(array)
            self.stds.append(n)

    def load_from_file(self):
        self.load_from_directory()

    def load_from_directory(self):
        if self.save_mean:
            self.means.load_from_file()

        if self.save_minmax:
            self.mins.load_from_file()
            self.maxes.load_from_file()

        if self.save_std:
            self.stds.load_from_file()

    def graph(self, xes=None):
        import pylab

        if self.save_minmax:
            mn = numpy.array(self.mins.tolist())
            mx = numpy.array(self.maxes.tolist())
            if self.save_mean:
                y = numpy.array(self.means.tolist())
            else:
                y = (mn+mx) / 2

            above_y = mx - y
            below_y = y - mn

            if not xes:
                xes = numpy.arange(len(y))

            pylab.errorbar(x=xes, y=y, yerr=[below_y, above_y])

        elif self.save_mean:
            y = numpy.array(self.means.tolist())
            if not xes:
                xes = numpy.arange(len(y))

            pylab.plot(x=xes, y=y)


class SeriesMultiplexer():
    def __init__(self):
        self._series_dict = {}
        self._warned_for = {}

    def append(self, series_name, item):
        # if we don't have the series, just don't do anything
        if self._series_dict.has_key(series_name):
            s = self._series_dict[series_name]
            s.append(item)
        elif not self._warned_for.has_key(series_name):
            print "WARNING: SeriesMultiplexer called with unknown name ", series_name
            self._warned_for[series_name] = 1

    def append_list(self, series_name, items):
        if self._series_dict.has_key(series_name):
            s = self._series_dict[series_name]
            s.append_list(items)
        elif not self._warned_for.has_key(series_name):
            print "WARNING: SeriesMultiplexer called with unknown name ", series_name
            self._warned_for[series_name] = 1

    def add_series(self, series):
        if self._series_dict.has_key(series.name):
            raise Exception("A series with such a name already exists")
        self._series_dict[series.name] = series

class SeriesList():
    def __init__(self, num_elements, name, directory, series_constructor=BaseSeries):
        self._subseries = [None] * num_elements
        self.name = name

        for i in range(num_elements):
            newname = name + "." + str(i)
            self._subseries[i] = series_constructor(name=newname, directory=directory)

    def load_from_files(self):
        self.load_from_file()

    def load_from_file(self):
        for s in self._subseries:
            s.load_from_file()

    # no "append_list", this would get confusing
    def append(self, list_of_items):
        if len(list_of_items) != len(self._subseries):
            raise Exception("bad number of items, expected " + str(len(self._subseries)) + ", got " + str(len(list_of_items)))
        for i in range(len(list_of_items)):
            self._subseries[i].append(list_of_items[i])


# Just a shortcut
class ParamsArrayStats(SeriesList):
    def __init__(self, num_params_arrays, name, directory):
        cstr = BasicStatsSeries.series_constructor()

        SeriesList.__init__(self, num_elements=num_params_arrays,
                                name=name, directory=directory,
                                series_constructor=cstr)

# ------------------------
# Utilities to work with the series files from the command line

# "dumpf"
def dump_floats_file(filepath):
    print "Floats dump of ", filepath
    with open(filepath, "rb") as f:
        s = os.stat(filepath)
        size = s.st_size
        num = size / 4
        a = array.array('f')
        a.fromfile(f, num)
        print a.tolist()

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "dumpf":
        file = args[1]
        dump_floats_file(file)
    else:
        print "Bad arguments"

