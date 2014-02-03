#!/usr/bin/python
# coding: utf-8

from __future__ import with_statement

import jobman
from jobman import DD

from pylearn.io.seriestables import *
import tables



# from pylearn codebase
# useful in __init__(param1, param2, etc.) to save
# values in self.param1, self.param2... just call
# update_locals(self, locals())
def update_locals(obj, dct):
    if 'self' in dct:
        del dct['self']
    obj.__dict__.update(dct)

# from a dictionary of possible values for hyperparameters, e.g.
# hp_values = {'learning_rate':[0.1, 0.01], 'num_layers': [1,2]}
# create a list of other dictionaries representing all the possible
# combinations, thus in this example creating:
# [{'learning_rate': 0.1, 'num_layers': 1}, ...]
# (similarly for combinations (0.1, 2), (0.01, 1), (0.01, 2))
def produit_cartesien_jobs(val_dict):
    job_list = [DD()]
    all_keys = val_dict.keys()

    for key in all_keys:
        possible_values = val_dict[key]
        new_job_list = []
        for val in possible_values:
            for job in job_list:
                to_insert = job.copy()
                to_insert.update({key: val})
                new_job_list.append(to_insert)
        job_list = new_job_list

    return job_list

def jobs_from_reinsert_list(cols, job_vals):
    job_list = []
    for vals in job_vals:
        job = DD()
        for i, col in enumerate(cols):
            job[col] = vals[i]
        job_list.append(job)

    return job_list

def save_params(all_params, filename):
    import pickle
    with open(filename, 'wb') as f:
        values = [p.value for p in all_params]

        # -1 for HIGHEST_PROTOCOL
        pickle.dump(values, f, -1)

# Perform insertion into the Postgre DB based on combination
# of hyperparameter values above
# (see comment for produit_cartesien_jobs() to know how it works)
def jobman_insert_job_vals(job_db, experiment_path, job_vals):
    jobs = produit_cartesien_jobs(job_vals)

    db = jobman.sql.db(job_db)
    for job in jobs:
        job.update({jobman.sql.EXPERIMENT: experiment_path})
        jobman.sql.insert_dict(job, db)

def jobman_insert_specific_jobs(job_db, experiment_path,
                        insert_cols, insert_vals):
    jobs = jobs_from_reinsert_list(insert_cols, insert_vals)

    db = jobman.sql.db(job_db)
    for job in jobs:
        job.update({jobman.sql.EXPERIMENT: experiment_path})
        jobman.sql.insert_dict(job, db)

# Just a shortcut for a common case where we need a few
# related Error (float) series
def get_accumulator_series_array( \
                hdf5_file, group_name, series_names, 
                reduce_every,
                index_names=('epoch','minibatch'),
                stdout_too=True,
                skip_hdf5_append=False):
    all_series = []

    new_group = hdf5_file.createGroup('/', group_name)

    other_targets = []
    if stdout_too:
        other_targets = [StdoutAppendTarget()]

    for sn in series_names:
        series_base = \
            ErrorSeries(error_name=sn,
                table_name=sn,
                hdf5_file=hdf5_file,
                hdf5_group=new_group._v_pathname,
                index_names=index_names,
                other_targets=other_targets,
                skip_hdf5_append=skip_hdf5_append)

        all_series.append( \
            AccumulatorSeriesWrapper( \
                    base_series=series_base,
                    reduce_every=reduce_every))

    ret_wrapper = SeriesArrayWrapper(all_series)

    return ret_wrapper


