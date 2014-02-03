import tables

import numpy
import time

##############################################################################
# Utility functions to create IsDescription objects (pytables data types)

'''
The way these "IsDescription constructor" work is simple: write the
code as if it were in a file, then exec()ute it, leaving us with
a local-scoped LocalDescription which may be used to call createTable.

It's a small hack, but it's necessary as the names of the columns
are retrieved based on the variable name, which we can't programmatically set
otherwise.
'''

def _get_description_timestamp_cpuclock_columns(store_timestamp, store_cpuclock, pos=0):
    toexec = ""

    if store_timestamp:
        toexec += "\ttimestamp = tables.Time32Col(pos="+str(pos)+")\n"
        pos += 1

    if store_cpuclock:
        toexec += "\tcpuclock = tables.Float64Col(pos="+str(pos)+")\n"
        pos += 1

    return toexec, pos

def _get_description_n_ints(int_names, int_width=64, pos=0):
    """
    Begins construction of a class inheriting from IsDescription
    to construct an HDF5 table with index columns named with int_names.

    See Series().__init__ to see how those are used.
    """
    int_constructor = "tables.Int64Col"
    if int_width == 32:
        int_constructor = "tables.Int32Col"
    elif not int_width in (32, 64):
        raise "int_width must be left unspecified, or should equal 32 or 64"

    toexec = ""

    for n in int_names:
        toexec += "\t" + n + " = " + int_constructor + "(pos=" + str(pos) + ")\n"
        pos += 1

    return toexec, pos

def _get_description_with_n_ints_n_floats(int_names, float_names, 
                        int_width=64, float_width=32,
                        store_timestamp=True, store_cpuclock=True):
    """
    Constructs a class to be used when constructing a table with PyTables.

    This is useful to construct a series with an index with multiple levels.
    E.g. if you want to index your "validation error" with "epoch" first, then
    "minibatch_index" second, you'd use two "int_names".

    Parameters
    ----------
    int_names : tuple of str
        Names of the int (e.g. index) columns
    float_names : tuple of str
        Names of the float (e.g. error) columns
    int_width : {'32', '64'}
        Type of ints.
    float_width : {'32', '64'}
        Type of floats.
    store_timestamp : bool
        See __init__ of Series
    store_cpuclock : bool
        See __init__ of Series

    Returns
    -------
    A class object, to pass to createTable()
    """

    toexec = "class LocalDescription(tables.IsDescription):\n"

    toexec_, pos = _get_description_timestamp_cpuclock_columns(store_timestamp, store_cpuclock)
    toexec += toexec_

    toexec_, pos = _get_description_n_ints(int_names, int_width=int_width, pos=pos)
    toexec += toexec_

    float_constructor = "tables.Float32Col"
    if float_width == 64:
        float_constructor = "tables.Float64Col"
    elif not float_width in (32, 64):
        raise "float_width must be left unspecified, or should equal 32 or 64"

    for n in float_names:
        toexec += "\t" + n + " = " + float_constructor + "(pos=" + str(pos) + ")\n"
        pos += 1

    exec(toexec)

    return LocalDescription

##############################################################################
# Series classes

# Shortcut to allow passing a single int as index, instead of a tuple
def _index_to_tuple(index):
    if type(index) == tuple:
        return index

    if type(index) == list:
        index = tuple(index)
        return index

    try:
        if index % 1 > 0.001 and index % 1 < 0.999:
            raise
        idx = long(index)
        return (idx,)
    except:
        raise TypeError("index must be a tuple of integers, or at least a single integer")

class Series():
    """
    Base Series class, with minimal arguments and type checks. 

    Yet cannot be used by itself (it's append() method raises an error)
    """

    def __init__(self, table_name, hdf5_file, index_names=('epoch',), 
                    title="", hdf5_group='/', 
                    store_timestamp=True, store_cpuclock=True):
        """Basic arguments each Series must get.

        Parameters
        ----------
        table_name : str
            Name of the table to create under group "hd5_group" (other 
            parameter). No spaces, ie. follow variable naming restrictions.
        hdf5_file : open HDF5 file
            File opened with openFile() in PyTables (ie. return value of 
            openFile).
        index_names : tuple of str
            Columns to use as index for elements in the series, other 
            example would be ('epoch', 'minibatch'). This would then allow
            you to call append(index, element) with index made of two ints,
            one for epoch index, one for minibatch index in epoch.
        title : str
            Title to attach to this table as metadata. Can contain spaces 
            and be longer then the table_name.
        hdf5_group : str
            Path of the group (kind of a file) in the HDF5 file under which
            to create the table.
        store_timestamp : bool
            Whether to create a column for timestamps and store them with 
            each record.
        store_cpuclock : bool
            Whether to create a column for cpu clock and store it with 
            each record.
        """

        #########################################
        # checks

        if type(table_name) != str:
            raise TypeError("table_name must be a string")
        if table_name == "":
            raise ValueError("table_name must not be empty")

        if not isinstance(hdf5_file, tables.file.File):
            raise TypeError("hdf5_file must be an open HDF5 file (use tables.openFile)")
        #if not ('w' in hdf5_file.mode or 'a' in hdf5_file.mode):
        #    raise ValueError("hdf5_file must be opened in write or append mode")

        if type(index_names) != tuple:
            raise TypeError("index_names must be a tuple of strings." + \
                    "If you have only one element in the tuple, don't forget " +\
                    "to add a comma, e.g. ('epoch',).")
        for name in index_names:
            if type(name) != str:
                raise TypeError("index_names must only contain strings, but also"+\
                        "contains a "+str(type(name))+".")

        if type(title) != str:
            raise TypeError("title must be a string, even if empty")

        if type(hdf5_group) != str:
            raise TypeError("hdf5_group must be a string")

        if type(store_timestamp) != bool:
            raise TypeError("store_timestamp must be a bool")

        if type(store_cpuclock) != bool:
            raise TypeError("store_timestamp must be a bool")

        #########################################

        self.table_name = table_name
        self.hdf5_file = hdf5_file
        self.index_names = index_names
        self.title = title
        self.hdf5_group = hdf5_group

        self.store_timestamp = store_timestamp
        self.store_cpuclock = store_cpuclock

    def append(self, index, element):
        raise NotImplementedError

    def _timestamp_cpuclock(self, newrow):
        if self.store_timestamp:
            newrow["timestamp"] = time.time()

        if self.store_cpuclock:
            newrow["cpuclock"] = time.clock()

class DummySeries():
    """
    To put in a series dictionary instead of a real series, to do nothing
    when we don't want a given series to be saved.

    E.g. if we'd normally have a "training_error" series in a dictionary
    of series, the training loop would have something like this somewhere:

        series["training_error"].append((15,), 20.0)

    but if we don't want to save the training errors this time, we simply
    do

        series["training_error"] = DummySeries()
    """
    def append(self, index, element):
        pass

class ErrorSeries(Series):
    """
    Most basic Series: saves a single float (called an Error as this is
    the most common use case I foresee) along with an index (epoch, for
    example) and timestamp/cpu.clock for each of these floats.
    """

    def __init__(self, error_name, table_name, 
                    hdf5_file, index_names=('epoch',), 
                    title="", hdf5_group='/', 
                    store_timestamp=True, store_cpuclock=True):
        """
        For most parameters, see Series.__init__

        Parameters
        ----------
        error_name : str
            In the HDF5 table, column name for the error float itself.
        """

        # most type/value checks are performed in Series.__init__
        Series.__init__(self, table_name, hdf5_file, index_names, title, 
                            store_timestamp=store_timestamp,
                            store_cpuclock=store_cpuclock)

        if type(error_name) != str:
            raise TypeError("error_name must be a string")
        if error_name == "":
            raise ValueError("error_name must not be empty")

        self.error_name = error_name

        self._create_table()

    def _create_table(self):
       table_description = _get_description_with_n_ints_n_floats( \
                                  self.index_names, (self.error_name,),
                                  store_timestamp=self.store_timestamp,
                                  store_cpuclock=self.store_cpuclock)

       self._table = self.hdf5_file.createTable(self.hdf5_group,
                            self.table_name, 
                            table_description,
                            title=self.title)


    def append(self, index, error):
        """
        Parameters
        ----------
        index : tuple of int
            Following index_names passed to __init__, e.g. (12, 15) if 
            index_names were ('epoch', 'minibatch_size').
            A single int (not tuple) is acceptable if index_names has a single 
            element.
            An array will be casted to a tuple, as a convenience.

        error : float
            Next error in the series.
        """
        index = _index_to_tuple(index)

        if len(index) != len(self.index_names):
            raise ValueError("index provided does not have the right length (expected " \
                            + str(len(self.index_names)) + " got " + str(len(index)))

        # other checks are implicit when calling newrow[..] =,
        # which should throw an error if not of the right type

        newrow = self._table.row

        # Columns for index in table are based on index_names
        for col_name, value in zip(self.index_names, index):
            newrow[col_name] = value
        newrow[self.error_name] = error

        # adds timestamp and cpuclock to newrow if necessary
        self._timestamp_cpuclock(newrow)

        newrow.append()

        self.hdf5_file.flush()

# Does not inherit from Series because it does not itself need to
# access the hdf5_file and does not need a series_name (provided
# by the base_series.)
class AccumulatorSeriesWrapper():
    '''
    Wraps a Series by accumulating objects passed its Accumulator.append()
    method and "reducing" (e.g. calling numpy.mean(list)) once in a while,
    every "reduce_every" calls in fact.
    '''

    def __init__(self, base_series, reduce_every, reduce_function=numpy.mean):
        """
        Parameters
        ----------
        base_series : Series
            This object must have an append(index, value) function.

        reduce_every : int
            Apply the reduction function (e.g. mean()) every time we get this 
            number of elements. E.g. if this is 100, then every 100 numbers 
            passed to append(), we'll take the mean and call append(this_mean) 
            on the BaseSeries.

        reduce_function : function
            Must take as input an array of "elements", as passed to (this 
            accumulator's) append(). Basic case would be to take an array of 
            floats and sum them into one float, for example.
        """
        self.base_series = base_series
        self.reduce_function = reduce_function
        self.reduce_every = reduce_every

        self._buffer = []

    
    def append(self, index, element):
        """
        Parameters
        ----------
        index : tuple of int
            The index used is the one of the last element reduced. E.g. if
            you accumulate over the first 1000 minibatches, the index
            passed to the base_series.append() function will be 1000.
            A single int (not tuple) is acceptable if index_names has a single 
            element.
            An array will be casted to a tuple, as a convenience.

        element : float
            Element that will be accumulated.
        """
        self._buffer.append(element)

        if len(self._buffer) == self.reduce_every:
            reduced = self.reduce_function(self._buffer)
            self.base_series.append(index, reduced)
            self._buffer = []

        # The >= case should never happen, except if lists
        # were appended by accessing _buffer externally (when it's
        # intended to be private), which should be a red flag.
        assert len(self._buffer) < self.reduce_every

# Outside of class to fix an issue with exec in Python 2.6.
# My sorries to the god of pretty code.
def _BasicStatisticsSeries_construct_table_toexec(index_names, store_timestamp, store_cpuclock):
    toexec = "class LocalDescription(tables.IsDescription):\n"

    toexec_, pos = _get_description_timestamp_cpuclock_columns(store_timestamp, store_cpuclock)
    toexec += toexec_

    toexec_, pos = _get_description_n_ints(index_names, pos=pos)
    toexec += toexec_

    toexec += "\tmean = tables.Float32Col(pos=" + str(pos) + ")\n"
    toexec += "\tmin = tables.Float32Col(pos=" + str(pos+1) + ")\n"
    toexec += "\tmax = tables.Float32Col(pos=" + str(pos+2) + ")\n"
    toexec += "\tstd = tables.Float32Col(pos=" + str(pos+3) + ")\n"
   
    # This creates "LocalDescription", which we may then use
    exec(toexec)

    return LocalDescription

# Defaults functions for BasicStatsSeries. These can be replaced.
_basic_stats_functions = {'mean': lambda(x): numpy.mean(x),
                    'min': lambda(x): numpy.min(x),
                    'max': lambda(x): numpy.max(x),
                    'std': lambda(x): numpy.std(x)}

class BasicStatisticsSeries(Series):
    
    def __init__(self, table_name, hdf5_file, 
                    stats_functions=_basic_stats_functions, 
                    index_names=('epoch',), title="", hdf5_group='/', 
                    store_timestamp=True, store_cpuclock=True):
        """
        For most parameters, see Series.__init__

        Parameters
        ----------
        series_name : str
            Not optional here. Will be prepended with "Basic statistics for "

        stats_functions : dict, optional
            Dictionary with a function for each key "mean", "min", "max", 
            "std". The function must take whatever is passed to append(...) 
            and return a single number (float).
        """

        # Most type/value checks performed in Series.__init__
        Series.__init__(self, table_name, hdf5_file, index_names, title, 
                            store_timestamp=store_timestamp,
                            store_cpuclock=store_cpuclock)

        if type(hdf5_group) != str:
            raise TypeError("hdf5_group must be a string")

        if type(stats_functions) != dict:
            # just a basic check. We'll suppose caller knows what he's doing.
            raise TypeError("stats_functions must be a dict")

        self.hdf5_group = hdf5_group

        self.stats_functions = stats_functions

        self._create_table()

    def _create_table(self):
        table_description = \
                _BasicStatisticsSeries_construct_table_toexec( \
                    self.index_names,
                    self.store_timestamp, self.store_cpuclock)

        self._table = self.hdf5_file.createTable(self.hdf5_group,
                         self.table_name, table_description)

    def append(self, index, array):
        """
        Parameters
        ----------
        index : tuple of int
            Following index_names passed to __init__, e.g. (12, 15) 
            if index_names were ('epoch', 'minibatch_size')
            A single int (not tuple) is acceptable if index_names has a single 
            element.
            An array will be casted to a tuple, as a convenience.

        array
            Is of whatever type the stats_functions passed to
            __init__ can take. Default is anything numpy.mean(),
            min(), max(), std() can take. 
        """
        index = _index_to_tuple(index)

        if len(index) != len(self.index_names):
            raise ValueError("index provided does not have the right length (expected " \
                            + str(len(self.index_names)) + " got " + str(len(index)))

        newrow = self._table.row

        for col_name, value in zip(self.index_names, index):
            newrow[col_name] = value

        newrow["mean"] = self.stats_functions['mean'](array)
        newrow["min"] = self.stats_functions['min'](array)
        newrow["max"] = self.stats_functions['max'](array)
        newrow["std"] = self.stats_functions['std'](array)

        self._timestamp_cpuclock(newrow)

        newrow.append()

        self.hdf5_file.flush()

class SeriesArrayWrapper():
    """
    Simply redistributes any number of elements to sub-series to respective 
    append()s.

    To use if you have many elements to append in similar series, e.g. if you 
    have an array containing [train_error, valid_error, test_error], and 3 
    corresponding series, this allows you to simply pass this array of 3 
    values to append() instead of passing each element to each individual 
    series in turn.
    """

    def __init__(self, base_series_list):
        """
        Parameters
        ----------
        base_series_list : array or tuple of Series
            You must have previously created and configured each of those
            series, then put them in an array. This array must follow the
            same order as the array passed as ``elements`` parameter of
            append().
        """
        self.base_series_list = base_series_list

    def append(self, index, elements):
        """
        Parameters
        ----------
        index : tuple of int
            See for example ErrorSeries.append()

        elements : array or tuple
            Array or tuple of elements that will be passed down to
            the base_series passed to __init__, in the same order.
        """
        if len(elements) != len(self.base_series_list):
            raise ValueError("not enough or too much elements provided (expected " \
                            + str(len(self.base_series_list)) + " got " + str(len(elements)))

        for series, el in zip(self.base_series_list, elements):
            series.append(index, el)

class SharedParamsStatisticsWrapper(SeriesArrayWrapper):
    '''
    Save mean, min/max, std of shared parameters place in an array.

    Here "shared" means "theano.shared", which means elements of the
    array will have a .value to use for numpy.mean(), etc.

    This inherits from SeriesArrayWrapper, which provides the append()
    method.
    '''

    def __init__(self, arrays_names, new_group_name, hdf5_file,
                    base_group='/', index_names=('epoch',), title="",
                    store_timestamp=True, store_cpuclock=True):
        """
        For other parameters, see Series.__init__

        Parameters
        ----------
        array_names : array or tuple of str
            Name of each array, in order of the array passed to append(). E.g. 
            ('layer1_b', 'layer1_W', 'layer2_b', 'layer2_W')

        new_group_name : str
            Name of a new HDF5 group which will be created under base_group to 
            store the new series.

        base_group : str
            Path of the group under which to create the new group which will
            store the series.

        title : str
            Here the title is attached to the new group, not a table.

        store_timestamp : bool
            Here timestamp and cpuclock are stored in *each* table

        store_cpuclock : bool
            Here timestamp and cpuclock are stored in *each* table
        """

        # most other checks done when calling BasicStatisticsSeries
        if type(new_group_name) != str:
            raise TypeError("new_group_name must be a string")
        if new_group_name == "":
            raise ValueError("new_group_name must not be empty")

        base_series_list = []

        new_group = hdf5_file.createGroup(base_group, new_group_name, title=title)

        stats_functions = {'mean': lambda(x): numpy.mean(x.value),
                    'min': lambda(x): numpy.min(x.value),
                    'max': lambda(x): numpy.max(x.value),
                    'std': lambda(x): numpy.std(x.value)}

        for name in arrays_names:
            base_series_list.append(
                        BasicStatisticsSeries(
                                table_name=name,
                                hdf5_file=hdf5_file,
                                index_names=index_names,
                                stats_functions=stats_functions,
                                hdf5_group=new_group._v_pathname,
                                store_timestamp=store_timestamp,
                                store_cpuclock=store_cpuclock))

        SeriesArrayWrapper.__init__(self, base_series_list)


