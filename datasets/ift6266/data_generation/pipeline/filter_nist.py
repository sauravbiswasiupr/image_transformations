import numpy
from pylearn.io import filetensor as ft
from ift6266 import datasets
from ift6266.datasets.ftfile import FTDataSet

dataset_str = 'P07_' # NISTP # 'P07safe_' 

#base_path = '/data/lisatmp/ift6266h10/data/'+dataset_str
#base_output_path = '/data/lisatmp/ift6266h10/data/transformed_digits/'+dataset_str+'train'

base_path = '/data/lisa/data/ift6266h10/data/'+dataset_str
base_output_path = '/data/lisatmp/ift6266h10/data/transformed_digits/'+dataset_str+'train'

for fileno in range(100):
    print "Processing file no ", fileno

    output_data_file = base_output_path+str(fileno)+'_data.ft'
    output_labels_file = base_output_path+str(fileno)+'_labels.ft'

    print "Reading from ",base_path+'train'+str(fileno)+'_data.ft'

    dataset = lambda maxsize=None, min_file=0, max_file=100: \
                    FTDataSet(train_data = [base_path+'train'+str(fileno)+'_data.ft'],
                       train_lbl = [base_path+'train'+str(fileno)+'_labels.ft'],
                       test_data = [base_path+'_test_data.ft'],
                       test_lbl = [base_path+'_test_labels.ft'],
                       valid_data = [base_path+'_valid_data.ft'],
                       valid_lbl = [base_path+'_valid_labels.ft'])
                       # no conversion or scaling... keep data as is
                       #indtype=theano.config.floatX, inscale=255., maxsize=maxsize)

    ds = dataset()

    all_x = []
    all_y = []

    all_count = 0

    for mb_x,mb_y in ds.train(1):
        if mb_y[0] <= 9:
            all_x.append(mb_x[0])
            all_y.append(mb_y[0])

        if (all_count+1) % 100000 == 0:
            print "Done next 100k"

        all_count += 1
   
    # data is stored as uint8 on 0-255
    merged_x = numpy.asarray(all_x, dtype=numpy.uint8)
    merged_y = numpy.asarray(all_y, dtype=numpy.int32)

    print "Kept", len(all_x), "(shape ", merged_x.shape, ") examples from", all_count

    f = open(output_data_file, 'wb')
    ft.write(f, merged_x)
    f.close()

    f = open(output_labels_file, 'wb')
    ft.write(f, merged_y)
    f.close()
    
