import numpy, Image
from pylearn.io import filetensor as ft


DATAPATH = '/data/lisa/data/'
DATASET = 'nist' # nist, p07, pnist
NUM_BATCHES = 250
BATCH_SIZE = 10
IMGSHP = (32,32)
WHITE_SPACE_THICKNESS = 1
DATASET_PATH = { 'nist' : [ DATAPATH + 'nist/by_class/all/all_test_data.ft',
                            DATAPATH + 'nist/by_class/all/all_test_labels.ft' ],
                 'p07'  : [ DATAPATH + 'ift6266h10/data/P07_test_data.ft',
                            DATAPATH + 'ift6266h10/data/P07_test_labels.ft' ],
                 'pnist': [ DATAPATH + 'ift6266h10/data/PNIST07_test_data.ft',
                            DATAPATH + 'ift6266h10/data/PNIST07_test_labels.ft' ] }

def generate_batches():
# Generate a directory containing NUM_BATCHES of DATASET
    total = NUM_BATCHES * BATCH_SIZE

    # Create a matrix of random integers within the range
    # [0,lenght_dataset-1]  

    f = open(DATASET_PATH[DATASET][0])
    g = open(DATASET_PATH[DATASET][1])
    test_data = ft.read(f)
    test_labels = ft.read(g)

    resulting_data = numpy.zeros((total,IMGSHP[0]*IMGSHP[1]))
    resulting_labels = numpy.zeros((total,))
    f.close();g.close()

    ds_size = len(test_data)
    rand_seq = numpy.random.random_integers(ds_size-1, size=(NUM_BATCHES,BATCH_SIZE))

    for i in range(NUM_BATCHES):
       for j in range(BATCH_SIZE):
           resulting_data[i*BATCH_SIZE+j]=test_data[rand_seq[i,j]]
           resulting_labels[i*BATCH_SIZE+j] = test_labels[rand_seq[i,j]]
       image = generate_image(resulting_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
       text = generate_labels(resulting_labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE], rand_seq[i])
       filename = DATASET + '_' + str("%04d" % int(i+1))
       image.save(filename+'.jpeg')
       save_text(text,filename)

    ft_name = 'AMT_'+DATASET+'_'+str(NUM_BATCHES)
    generate_ft_file(resulting_data,resulting_labels,ft_name)

def save_text(text,filename):
    f = open(filename+'.txt', 'w')
    f.write(text)
    f.close()
def generate_ft_file(data,labels,ft_name):
    fdata = open(ft_name+'_data.ft','w')
    flabels = open(ft_name+'_labels.ft','w')
    ft.write(fdata,data)
    ft.write(flabels,labels)
    fdata.close();flabels.close()

def generate_image(seq):
    all_images = []

    white_space = numpy.asarray(numpy.zeros((IMGSHP[0],WHITE_SPACE_THICKNESS))+255.,dtype='uint8')

    for i in range(len(seq)):
        all_images += [numpy.asarray(seq[i].reshape((IMGSHP)),dtype='uint8')]

    all_images_stacked = numpy.hstack(numpy.asarray([numpy.hstack((image,white_space)) for image in all_images]))
    return Image.fromarray(all_images_stacked)

def generate_labels(seq, indexes):
    return str(seq) + '\n' + str(indexes)

if __name__ =='__main__':
    print 'Starting data generation'
    generate_batches()
