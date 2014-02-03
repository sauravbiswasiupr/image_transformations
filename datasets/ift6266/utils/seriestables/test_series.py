import tempfile

import numpy
import numpy.random

from jobman import DD

import tables

from series import *
import series

#################################################
# Utils

def compare_floats(f1,f2):
    if f1-f2 < 1e-3:
        return True
    return False

def compare_lists(it1, it2, floats=False):
    if len(it1) != len(it2):
        return False

    for el1,  el2 in zip(it1, it2):
        if floats:
            if not compare_floats(el1,el2):
                return False
        elif el1 != el2:
            return False

    return True

#################################################
# Basic Series class tests

def test_Series_types():
    pass

#################################################
# ErrorSeries tests

def test_ErrorSeries_common_case(h5f=None):
    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    validation_error = series.ErrorSeries(error_name="validation_error", table_name="validation_error",
                                hdf5_file=h5f, index_names=('epoch','minibatch'),
                                title="Validation error indexed by epoch and minibatch")

    # (1,1), (1,2) etc. are (epoch, minibatch) index
    validation_error.append((1,1), 32.0)
    validation_error.append((1,2), 30.0)
    validation_error.append((2,1), 28.0)
    validation_error.append((2,2), 26.0)

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")
    
    table = h5f.getNode('/', 'validation_error')

    assert compare_lists(table.cols.epoch[:], [1,1,2,2])
    assert compare_lists(table.cols.minibatch[:], [1,2,1,2])
    assert compare_lists(table.cols.validation_error[:], [32.0, 30.0, 28.0, 26.0])

def test_ErrorSeries_no_index(h5f=None):
    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    validation_error = series.ErrorSeries(error_name="validation_error",
                                table_name="validation_error",
                                hdf5_file=h5f, 
                                # empty tuple
                                index_names=tuple(),
                                title="Validation error with no index")

    # (1,1), (1,2) etc. are (epoch, minibatch) index
    validation_error.append(tuple(), 32.0)
    validation_error.append(tuple(), 30.0)
    validation_error.append(tuple(), 28.0)
    validation_error.append(tuple(), 26.0)

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")
    
    table = h5f.getNode('/', 'validation_error')

    assert compare_lists(table.cols.validation_error[:], [32.0, 30.0, 28.0, 26.0])
    assert not ("epoch" in dir(table.cols))

def test_ErrorSeries_notimestamp(h5f=None):
    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    validation_error = series.ErrorSeries(error_name="validation_error", table_name="validation_error",
                                hdf5_file=h5f, index_names=('epoch','minibatch'),
                                title="Validation error indexed by epoch and minibatch", 
                                store_timestamp=False)

    # (1,1), (1,2) etc. are (epoch, minibatch) index
    validation_error.append((1,1), 32.0)

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")
    
    table = h5f.getNode('/', 'validation_error')

    assert compare_lists(table.cols.epoch[:], [1])
    assert not ("timestamp" in dir(table.cols))
    assert "cpuclock" in dir(table.cols)

def test_ErrorSeries_nocpuclock(h5f=None):
    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    validation_error = series.ErrorSeries(error_name="validation_error", table_name="validation_error",
                                hdf5_file=h5f, index_names=('epoch','minibatch'),
                                title="Validation error indexed by epoch and minibatch", 
                                store_cpuclock=False)

    # (1,1), (1,2) etc. are (epoch, minibatch) index
    validation_error.append((1,1), 32.0)

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")
    
    table = h5f.getNode('/', 'validation_error')

    assert compare_lists(table.cols.epoch[:], [1])
    assert not ("cpuclock" in dir(table.cols))
    assert "timestamp" in dir(table.cols)

def test_AccumulatorSeriesWrapper_common_case(h5f=None):
    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    validation_error = ErrorSeries(error_name="accumulated_validation_error",
                                table_name="accumulated_validation_error",
                                hdf5_file=h5f,
                                index_names=('epoch','minibatch'),
                                title="Validation error, summed every 3 minibatches, indexed by epoch and minibatch")

    accumulator = AccumulatorSeriesWrapper(base_series=validation_error,
                                    reduce_every=3, reduce_function=numpy.sum)

    # (1,1), (1,2) etc. are (epoch, minibatch) index
    accumulator.append((1,1), 32.0)
    accumulator.append((1,2), 30.0)
    accumulator.append((2,1), 28.0)
    accumulator.append((2,2), 26.0)
    accumulator.append((3,1), 24.0)
    accumulator.append((3,2), 22.0)

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")
    
    table = h5f.getNode('/', 'accumulated_validation_error')

    assert compare_lists(table.cols.epoch[:], [2,3])
    assert compare_lists(table.cols.minibatch[:], [1,2])
    assert compare_lists(table.cols.accumulated_validation_error[:], [90.0,72.0], floats=True)

def test_BasicStatisticsSeries_common_case(h5f=None):
    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    stats_series = BasicStatisticsSeries(table_name="b_vector_statistics",
                                hdf5_file=h5f, index_names=('epoch','minibatch'),
                                title="Basic statistics for b vector indexed by epoch and minibatch")

    # (1,1), (1,2) etc. are (epoch, minibatch) index
    stats_series.append((1,1), [0.15, 0.20, 0.30])
    stats_series.append((1,2), [-0.18, 0.30, 0.58])
    stats_series.append((2,1), [0.18, -0.38, -0.68])
    stats_series.append((2,2), [0.15, 0.02, 1.9])

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")
    
    table = h5f.getNode('/', 'b_vector_statistics')

    assert compare_lists(table.cols.epoch[:], [1,1,2,2])
    assert compare_lists(table.cols.minibatch[:], [1,2,1,2])
    assert compare_lists(table.cols.mean[:], [0.21666667,  0.23333333, -0.29333332,  0.69], floats=True)
    assert compare_lists(table.cols.min[:], [0.15000001, -0.18000001, -0.68000001,  0.02], floats=True)
    assert compare_lists(table.cols.max[:], [0.30, 0.58, 0.18, 1.9], floats=True)
    assert compare_lists(table.cols.std[:], [0.06236095, 0.31382939,  0.35640177, 0.85724366], floats=True)

def test_SharedParamsStatisticsWrapper_commoncase(h5f=None):
    import numpy.random

    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    stats = SharedParamsStatisticsWrapper(new_group_name="params", base_group="/",
                                arrays_names=('b1','b2','b3'), hdf5_file=h5f,
                                index_names=('epoch','minibatch'))

    b1 = DD({'value':numpy.random.rand(5)})
    b2 = DD({'value':numpy.random.rand(5)})
    b3 = DD({'value':numpy.random.rand(5)})
    stats.append((1,1), [b1,b2,b3])

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")

    b1_table = h5f.getNode('/params', 'b1')
    b3_table = h5f.getNode('/params', 'b3')

    assert b1_table.cols.mean[0] - numpy.mean(b1.value) < 1e-3
    assert b3_table.cols.mean[0] - numpy.mean(b3.value) < 1e-3
    assert b1_table.cols.min[0] - numpy.min(b1.value) < 1e-3
    assert b3_table.cols.min[0] - numpy.min(b3.value) < 1e-3

def test_SharedParamsStatisticsWrapper_notimestamp(h5f=None):
    import numpy.random

    if not h5f:
        h5f_path = tempfile.NamedTemporaryFile().name
        h5f = tables.openFile(h5f_path, "w")

    stats = SharedParamsStatisticsWrapper(new_group_name="params", base_group="/",
                                arrays_names=('b1','b2','b3'), hdf5_file=h5f,
                                index_names=('epoch','minibatch'),
                                store_timestamp=False)

    b1 = DD({'value':numpy.random.rand(5)})
    b2 = DD({'value':numpy.random.rand(5)})
    b3 = DD({'value':numpy.random.rand(5)})
    stats.append((1,1), [b1,b2,b3])

    h5f.close()

    h5f = tables.openFile(h5f_path, "r")

    b1_table = h5f.getNode('/params', 'b1')
    b3_table = h5f.getNode('/params', 'b3')

    assert b1_table.cols.mean[0] - numpy.mean(b1.value) < 1e-3
    assert b3_table.cols.mean[0] - numpy.mean(b3.value) < 1e-3
    assert b1_table.cols.min[0] - numpy.min(b1.value) < 1e-3
    assert b3_table.cols.min[0] - numpy.min(b3.value) < 1e-3

    assert not ('timestamp' in dir(b1_table.cols))

def test_get_desc():
    h5f_path = tempfile.NamedTemporaryFile().name
    h5f = tables.openFile(h5f_path, "w")

    desc = series._get_description_with_n_ints_n_floats(("col1","col2"), ("col3","col4"))

    mytable = h5f.createTable('/', 'mytable', desc)

    # just make sure the columns are there... otherwise this will throw an exception
    mytable.cols.col1
    mytable.cols.col2
    mytable.cols.col3
    mytable.cols.col4

    try:
        # this should fail... LocalDescription must be local to get_desc_etc
        test = LocalDescription
        assert False
    except:
        assert True

    assert True

def test_index_to_tuple_floaterror():
    try:
        series._index_to_tuple(5.1)
        assert False
    except TypeError:
        assert True

def test_index_to_tuple_arrayok():
    tpl = series._index_to_tuple([1,2,3])
    assert type(tpl) == tuple and tpl[1] == 2 and tpl[2] == 3

def test_index_to_tuple_intbecomestuple():
    tpl = series._index_to_tuple(32)

    assert type(tpl) == tuple and tpl == (32,)

def test_index_to_tuple_longbecomestuple():
    tpl = series._index_to_tuple(928374928374928L)

    assert type(tpl) == tuple and tpl == (928374928374928L,)

if __name__ == '__main__':
    import tempfile
    test_get_desc()
    test_ErrorSeries_common_case()
    test_BasicStatisticsSeries_common_case()
    test_AccumulatorSeriesWrapper_common_case()
    test_SharedParamsStatisticsWrapper_commoncase()

