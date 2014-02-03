#!/usr/bin/python
# coding: utf-8

from __future__ import with_statement

from jobman import DD

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

def test_produit_cartesien_jobs():
    vals = {'a': [1,2], 'b': [3,4,5]}
    print produit_cartesien_jobs(vals)


# taken from http://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
"""Simple module for getting amount of memory used by a specified user's
processes on a UNIX system.
It uses UNIX ps utility to get the memory usage for a specified username and
pipe it to awk for summing up per application memory usage and return the total.
Python's Popen() from subprocess module is used for spawning ps and awk.

"""

import subprocess

class MemoryMonitor(object):

    def __init__(self, username):
        """Create new MemoryMonitor instance."""
        self.username = username

    def usage(self):
        """Return int containing memory used by user's processes."""
        self.process = subprocess.Popen("ps -u %s -o rss | awk '{sum+=$1} END {print sum}'" % self.username,
                                        shell=True,
                                        stdout=subprocess.PIPE,
                                        )
        self.stdout_list = self.process.communicate()[0].split('\n')
        return int(self.stdout_list[0])

