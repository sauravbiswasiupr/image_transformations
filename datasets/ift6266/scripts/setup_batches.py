# -*- coding: utf-8 -*-

import random
from numpy import *
from pylearn.io import filetensor as ft

class Batches():
  def __init__(self):
    data_path = '/data/lisa/data/nist/by_class/'

    digits_train_data = 'digits/digits_train_data.ft'
    digits_train_labels = 'digits/digits_train_labels.ft'
    digits_test_data = 'digits/digits_test_data.ft'
    digits_test_labels = 'digits/digits_test_labels.ft'

    lower_train_data = 'lower/lower_train_data.ft'
    lower_train_labels = 'lower/lower_train_labels.ft'
    lower_test_data = 'lower/lower_test_data.ft'
    lower_test_labels = 'lower/lower_test_labels.ft'

    upper_train_data = 'upper/upper_train_data.ft'
    upper_train_labels = 'upper/upper_train_labels.ft'
    upper_test_data = 'upper/upper_test_data.ft'
    upper_test_labels = 'upper/upper_test_labels.ft'

    test_data = 'all/all_test_data.ft'
    test_labels = 'all/all_test_labels.ft'

    print 'Opening data...'

    f_digits_train_data = open(data_path + digits_train_data)
    f_digits_train_labels = open(data_path + digits_train_labels)
    f_digits_test_data = open(data_path + digits_test_data)
    f_digits_test_labels = open(data_path + digits_test_labels)

    f_lower_train_data = open(data_path + lower_train_data)
    f_lower_train_labels = open(data_path + lower_train_labels)
    f_lower_test_data = open(data_path + lower_test_data)
    f_lower_test_labels = open(data_path + lower_test_labels)

    f_upper_train_data = open(data_path + upper_train_data)
    f_upper_train_labels = open(data_path + upper_train_labels)
    f_upper_test_data = open(data_path + upper_test_data)
    f_upper_test_labels = open(data_path + upper_test_labels)

    #f_test_data = open(data_path + test_data)
    #f_test_labels = open(data_path + test_labels)

    self.raw_digits_train_data = ft.read(f_digits_train_data)
    self.raw_digits_train_labels = ft.read(f_digits_train_labels)
    self.raw_digits_test_data = ft.read(f_digits_test_data)
    self.raw_digits_test_labels = ft.read(f_digits_test_labels)

    self.raw_lower_train_data = ft.read(f_lower_train_data)
    self.raw_lower_train_labels = ft.read(f_lower_train_labels)
    self.raw_lower_test_data = ft.read(f_lower_test_data)
    self.raw_lower_test_labels = ft.read(f_lower_test_labels)

    self.raw_upper_train_data = ft.read(f_upper_train_data)
    self.raw_upper_train_labels = ft.read(f_upper_train_labels)
    self.raw_upper_test_data = ft.read(f_upper_test_data)
    self.raw_upper_test_labels = ft.read(f_upper_test_labels)

    #self.raw_test_data = ft.read(f_test_data)
    #self.raw_test_labels = ft.read(f_test_labels)

    f_digits_train_data.close()
    f_digits_train_labels.close()
    f_digits_test_data.close()
    f_digits_test_labels.close()

    f_lower_train_data.close()
    f_lower_train_labels.close()
    f_lower_test_data.close()
    f_lower_test_labels.close()

    f_upper_train_data.close()
    f_upper_train_labels.close()
    f_upper_test_data.close()
    f_upper_test_labels.close()

    #f_test_data.close()
    #f_test_labels.close()

    print 'Data opened'

  def set_batches(self, main_class = "d", start_ratio = -1, end_ratio = -1, batch_size = 20, verbose = False):
    self.batch_size = batch_size

    digits_train_size = len(self.raw_digits_train_labels)
    digits_test_size = len(self.raw_digits_test_labels)

    lower_train_size = len(self.raw_lower_train_labels)

    upper_train_size = len(self.raw_upper_train_labels)
    upper_test_size = len(self.raw_upper_test_labels)

    if verbose == True:
      print 'digits_train_size = %d' %digits_train_size
      print 'digits_test_size = %d' %digits_test_size
      print 'lower_train_size = %d' %lower_train_size
      print 'upper_train_size = %d' %upper_train_size
      print 'upper_test_size = %d' %upper_test_size

    if main_class == "u":
	# define main and other datasets
	raw_main_train_data = self.raw_upper_train_data
	raw_other_train_data1 = self.raw_lower_train_labels
	raw_other_train_data2 = self.raw_digits_train_labels
	raw_test_data = self.raw_upper_test_data

	raw_main_train_labels = self.raw_upper_train_labels
	raw_other_train_labels1 = self.raw_lower_train_labels
	raw_other_train_labels2 = self.raw_digits_train_labels
	raw_test_labels = self.raw_upper_test_labels

    elif main_class == "l":
	# define main and other datasets
	raw_main_train_data = self.raw_lower_train_data
	raw_other_train_data1 = self.raw_upper_train_labels
	raw_other_train_data2 = self.raw_digits_train_labels
	raw_test_data = self.raw_lower_test_data

	raw_main_train_labels = self.raw_lower_train_labels
	raw_other_train_labels1 = self.raw_upper_train_labels
	raw_other_train_labels2 = self.raw_digits_train_labels
	raw_test_labels = self.raw_lower_test_labels

    else:
	main_class = "d"
	# define main and other datasets
	raw_main_train_data = self.raw_digits_train_data
	raw_other_train_data1 = self.raw_lower_train_labels
	raw_other_train_data2 = self.raw_upper_train_labels
	raw_test_data = self.raw_digits_test_data

	raw_main_train_labels = self.raw_digits_train_labels
	raw_other_train_labels1 = self.raw_lower_train_labels
	raw_other_train_labels2 = self.raw_upper_train_labels
	raw_test_labels = self.raw_digits_test_labels

    main_train_size = len(raw_main_train_labels)
    other_train_size1 = len(raw_other_train_labels1)
    other_train_size2 = len(raw_other_train_labels2)
    other_train_size = other_train_size1 + other_train_size2

    test_size = len(raw_test_labels)
    test_size = int(test_size/batch_size)
    test_size *= batch_size
    validation_size = test_size 

    # default ratio is actual ratio
    if start_ratio == -1:
      self.start_ratio = float(main_train_size - test_size) / float(main_train_size + other_train_size)
    else:
      self.start_ratio = start_ratio

    if start_ratio == -1:
      self.end_ratio = float(main_train_size - test_size) / float(main_train_size + other_train_size)
    else:
      self.end_ratio = end_ratio

    if verbose == True:
      print 'main class : %s' %main_class
      print 'start_ratio = %f' %self.start_ratio
      print 'end_ratio = %f' %self.end_ratio

    i_main = 0
    i_other1 = 0
    i_other2 = 0
    i_batch = 0

    # compute the number of batches given start and end ratios
    n_main_batch = (main_train_size - test_size - batch_size * (self.end_ratio - self.start_ratio) / 2 ) / (batch_size * (self.start_ratio + (self.end_ratio - self.start_ratio) / 2))
    if (batch_size != batch_size * (self.start_ratio + (self.end_ratio - self.start_ratio) / 2)):
      n_other_batch = (other_train_size - batch_size * (self.end_ratio - self.start_ratio) / 2 ) / (batch_size - batch_size * (self.start_ratio + (self.end_ratio - self.start_ratio) / 2))
    else:
      n_other_batch = n_main_batch

    n_batches = min([n_main_batch, n_other_batch])

    # train batches
    self.train_batches = []

    # as long as we have data left in main and other, we create batches
    while i_main < main_train_size - batch_size - test_size and i_other1 < other_train_size1 - batch_size and i_other2 < other_train_size2 - batch_size:
      ratio = self.start_ratio + i_batch * (self.end_ratio - self.start_ratio) / n_batches
      batch_data = copy(raw_main_train_data[0:self.batch_size])
      batch_labels = copy(raw_main_train_labels[0:self.batch_size])

      for i in xrange(0, self.batch_size): # randomly choose between main and other, given the current ratio
	rnd1 = random.randint(0, 100)

	if rnd1 < 100 * ratio:
	  batch_data[i] = raw_main_train_data[i_main]
	  batch_labels[i] = raw_main_train_labels[i_main]
	  i_main += 1
	else:
	  rnd2 = random.randint(0, 100)

	  if rnd2 < 100 * float(other_train_size1) / float(other_train_size):
	    batch_data[i] = raw_other_train_data1[i_other1]
	    batch_labels[i] = raw_other_train_labels1[i_other1]
	    i_other1 += 1
	  else:
	    batch_data[i] = raw_other_train_data2[i_other2]
	    batch_labels[i] = raw_other_train_labels2[i_other2]
	    i_other2 += 1

      self.train_batches = self.train_batches + \
	      [(batch_data, batch_labels)]
      i_batch += 1

    offset = i_main

    # test batches
    self.test_batches = []
    for i in xrange(0, test_size, batch_size):
        self.test_batches = self.test_batches + \
            [(raw_test_data[i:i+batch_size], raw_test_labels[i:i+batch_size])]

    # validation batches
    self.validation_batches = []
    for i in xrange(0, validation_size, batch_size):
        self.validation_batches = self.validation_batches + \
            [(raw_main_train_data[offset+i:offset+i+batch_size], raw_main_train_labels[offset+i:offset+i+batch_size])]

    if verbose == True:
      print 'n_main = %d' %i_main
      print 'n_other1 = %d' %i_other1
      print 'n_other2 = %d' %i_other2
      print 'nb_train_batches = %d / %d' %(i_batch,n_batches)
      print 'offset = %d' %offset

  def get_train_batches(self):
    return self.train_batches

  def get_test_batches(self):
    return self.test_batches

  def get_validation_batches(self):
    return self.validation_batches

  def test_set_batches(self, intervall = 1000):
    for i in xrange(0, len(self.train_batches) - self.batch_size, intervall):
	n_main = 0

	for j in xrange(0, self.batch_size):
	  if self.train_batches[i][1][j] < 10:
	    n_main +=1
	print 'ratio batch %d : %f' %(i,float(n_main) / float(self.batch_size))

if __name__ == '__main__':
    batches = Batches()
    batches.set_batches(0.5,1, 20, True)
    batches.test_set_batches()
