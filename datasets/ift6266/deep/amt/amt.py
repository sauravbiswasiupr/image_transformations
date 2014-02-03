# Script usage : python amt.py filname.cvs type
"""
[rifaisal@timide ../fix/ift6266/deep/amt]$ python amt.py pnist.csv all
Testing on         : all
Total entries      : 300.0
Turks per batch    : 3
Average test error : 45.3333333333%
Error variance     : 7.77777777778%
[rifaisal@timide ../fix/ift6266/deep/amt]$ python amt.py pnist.csv 36
Testing on         : 36
Total entries      : 300.0
Turks per batch    : 3
Average test error : 51.6666666667%
Error variance     : 3.33333333333%
[rifaisal@timide ../fix/ift6266/deep/amt]$ python amt.py pnist.csv upper
Testing on         : upper
Total entries      : 63.0
Turks per batch    : 3
Average test error : 53.9682539683%
Error variance     : 1.77777777778%
[rifaisal@timide ../fix/ift6266/deep/amt]$ python amt.py pnist.csv lower
Testing on         : lower
Total entries      : 135.0
Turks per batch    : 3
Average test error : 37.037037037%
Error variance     : 3.77777777778%
[rifaisal@timide ../fix/ift6266/deep/amt]$ python amt.py pnist.csv digits
Testing on         : digits
Total entries      : 102.0
Turks per batch    : 3
Average test error : 50.9803921569%
Error variance     : 1.33333333333%
"""

import csv,numpy,re,decimal
from ift6266 import datasets
from pylearn.io import filetensor as ft

fnist = open('nist_train_class_freq.ft','r')
fp07 = open('p07_train_class_freq.ft','r')
fpnist = open('pnist_train_class_freq.ft','r')

nist_freq_table = ft.read(fnist)
p07_freq_table  = ft.read(fp07)
pnist_freq_table  = ft.read(fpnist)

fnist.close();fp07.close();fpnist.close()

DATASET_PATH = { 'nist' : '/data/lisa/data/ift6266h10/amt_data/nist/',
                 'p07' : '/data/lisa/data/ift6266h10/amt_data/p07/',
                 'pnist' : '/data/lisa/data/ift6266h10/amt_data/pnist/' }

freq_tables = { 'nist' : nist_freq_table,
                'p07'  : p07_freq_table,
                'pnist': pnist_freq_table }


CVSFILE = None
#PATH = None
answer_labels = [ 'Answer.c'+str(i+1) for i in range(10) ]
img_url = 'Input.image_url'
turks_per_batch = 3
image_per_batch = 10
TYPE = None

def all_classes_assoc():
    answer_assoc = {}
    for i in range(0,10):
        answer_assoc[str(i)]=i
    for i in range(10,36):
        answer_assoc[chr(i+55)]=i
    for i in range(36,62):
        answer_assoc[chr(i+61)]=i
    return answer_assoc

def upper_classes_assoc():
    answer_assoc = {}
    for i in range(10,36):
        answer_assoc[chr(i+55)]=i
    return answer_assoc

def lower_classes_assoc():
    answer_assoc = {}
    for i in range(36,62):
        answer_assoc[chr(i+61)]=i
    return answer_assoc

def digit_classes_assoc():
    answer_assoc = {}
    for i in range(0,10):
        answer_assoc[str(i)]=i
    return answer_assoc

def tsix_classes_assoc():
    answer_assoc = {}
    for i in range(0,10):
        answer_assoc[str(i)]=i
    for i in range(10,36):
        answer_assoc[chr(i+55)]=i
        answer_assoc[chr(i+87)]=i
    return answer_assoc

def upper_label_assoc(ulabel):
    for i in range(len(ulabel)):
        if ulabel[i] < 10 or ulabel[i] > 35 :
            ulabel[i] = -1
    return ulabel

def lower_label_assoc(ulabel):
    for i in range(len(ulabel)):
        if ulabel[i] < 36 or ulabel[i] > 61 :
            ulabel[i] = -1
    return ulabel

def tsix_label_assoc(ulabel): 
    for i in range(len(ulabel)):
        if ulabel[i] > 35 and ulabel[i] < 62 :
            ulabel[i] = ulabel[i] - 26
    return ulabel

def digit_label_assoc(ulabel):
    for i in range(len(ulabel)):
        if ulabel[i] < 0 or ulabel[i] > 9 :
            ulabel[i] = -1

    return ulabel

def classes_answer(type):
    if type == 'all':
        return all_classes_assoc()
    elif type == '36':
        return tsix_classes_assoc()
    elif type == 'lower':
        return lower_classes_assoc()
    elif type == 'upper':
        return upper_classes_assoc()
    elif type == 'digits':
        return digit_classes_assoc()
    else:
        raise ('Inapropriate option for the type of classification :' + type)


def test_error(assoc_type=TYPE,consensus=True):
    answer_assoc = classes_answer(assoc_type)

    turks = []
    reader = csv.DictReader(open(CVSFILE), delimiter=',')
    entries = [ turk for turk in reader ]

    errors = numpy.zeros((len(entries),))
    if len(entries) % turks_per_batch != 0 :
        raise Exception('Wrong number of entries or turks_per_batch')

    total_uniq_entries = len(entries) / turks_per_batch


    error_variances = numpy.zeros((total_uniq_entries,))
    
    if consensus:
        errors = numpy.zeros((total_uniq_entries,))
        num_examples = numpy.zeros((total_uniq_entries,))
        for i in range(total_uniq_entries):
            errors[i],num_examples[i] = get_turk_consensus_error(entries[i*turks_per_batch:(i+1)*turks_per_batch],assoc_type)
            error_variances[i] = errors[i*turks_per_batch:(i+1)*turks_per_batch].var()
    else:
        errors = numpy.zeros((len(entries),))
        num_examples = numpy.zeros((len(entries),))
        for i in range(total_uniq_entries):
            for t in range(turks_per_batch):
                errors[i*turks_per_batch+t],num_examples[i*turks_per_batch+t] = get_error(entries[i*turks_per_batch+t],assoc_type)
            error_variances[i] = errors[i*turks_per_batch:(i+1)*turks_per_batch].var()
        
    percentage_error = 100. * errors.sum() / num_examples.sum()
    print 'Testing on         : ' + str(assoc_type)
    print 'Total entries      : ' + str(num_examples.sum())
    print 'Turks per batch    : ' + str(turks_per_batch)
    print 'Average test error : ' + str(percentage_error) +'%'
    print 'Error variance     : ' + str(error_variances.mean()*image_per_batch) +'%' 


def find_dataset(entry):
    file = parse_filename(entry[img_url])
    return file.split('_')[0]
    
def get_error(entry, type):
    answer_assoc = classes_answer(type)
    labels = get_labels(entry,type)
    test_error = 0
    cnt = 0
    for i in range(len(answer_labels)):
        if labels[i] == -1:
            cnt+=1
        else:
            answer = entry[answer_labels[i]]
            try:
                if answer_assoc[answer] != labels[i]:
                    test_error+=1
            except:
                test_error+=1
    return test_error,image_per_batch-cnt

def get_turk_consensus_error(entries, type):
    answer_assoc = classes_answer(type)
    labels = get_labels(entries[0],type)
    test_error = 0
    cnt = 0
    answer= []
    freq_t = freq_tables[find_dataset(entries[0])]
    for i in range(len(answer_labels)):
        if labels[i] == -1:
            cnt+=1
        else:
            answers = [ entry[answer_labels[i]] for entry in entries ]
            if answers[0] != answers[1] and answers[1] != answers[2] and answers[0] != answers[2]:
                m = max([ freq_t[answer_assoc[answer]] for answer in answers])
                for answer in answers:
                    if freq_t[answer_assoc[answer]] == m :
                        a = answer
            else:
                for answer in answers:
                    if answers.count(answer) > 1 :
                        a =answer
            try:
                if answer_assoc[answer] != labels[i]:
                    test_error+=1
            except:
                test_error+=1
    return test_error,image_per_batch-cnt
def frequency_table():
    filenames = ['nist_train_class_freq.ft','p07_train_class_freq.ft','pnist_train_class_freq.ft']
    iterators = [datasets.nist_all(),datasets.nist_P07(),datasets.PNIST07()]
    for dataset,filename in zip(iterators,filenames):
        freq_table = numpy.zeros(62)
        for x,y in dataset.train(1):
            freq_table[int(y)]+=1
        f = open(filename,'w')
        ft.write(f,freq_table)
        f.close()

def get_labels(entry,type):
    file = parse_filename(entry[img_url])
    path = DATASET_PATH[find_dataset(entry)]
    f = open(path+file,'r')
    str_labels = re.sub("\s+", "",f.readline()).strip()[1:-2].split('.')
    unrestricted_labels = [ int(element) for element in str_labels ]
    if type == 'all':
        return unrestricted_labels
    elif type == '36':
        return tsix_label_assoc(unrestricted_labels)
    elif type == 'lower':
        return lower_label_assoc(unrestricted_labels)
    elif type == 'upper':
        return upper_label_assoc(unrestricted_labels)
    elif type == 'digits':
        return digit_label_assoc(unrestricted_labels)
    else:
        raise ('Inapropriate option for the type of classification :' + str(type))
    

def parse_filename(string):
    filename = string.split('/')[-1]
    return filename.split('.')[0]+'.txt'

if __name__ =='__main__':
    import sys
    CVSFILE = sys.argv[1]
    test_error(sys.argv[2],int(sys.argv[3]))
    #frequency_table()

