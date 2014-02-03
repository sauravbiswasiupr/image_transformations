import PIL
import numpy, cPickle, gzip
from numpy import *
from utils import *
#from logistic_sgd import load_data

def generate_img(source, n=1):

    
    #param : source = the path to the pickled data 
    #param : n = the number of tiled image to create 
    
    #loading the dataset contaning images and labels (we gonna sample from the test set)
    f = gzip.open(source,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    test_set_x, test_set_y = test_set
    f.close()

    max = test_set_x.shape[0]    

    # creating n images containing each 10 randomly pickes caraters from the test set

    for i in range(n):
 
        #picking randomly 10 images in the test set with their labels
        rng = numpy.random.RandomState(None)
        sample_idx = rng.randint(max)
        samples = numpy.array(test_set_x[sample_idx: sample_idx + 10 ])
        samples_labels = numpy.array(test_set_y[sample_idx: sample_idx + 10 ])


        #tiling images into a PIL images and saving it
        image = PIL.Image.fromarray(tile_raster_images( samples,
                     img_shape = (28,28), tile_shape = (1,10), 
                     tile_spacing=(1,1)))

        print ' ... is saving images'
        img_name = source + str(i)
        image.save(img_name+'.png')

        #saving the corresponding labels : todo after testing if the saving works
        print '... is saving labels'
        numpy.savetxt(img_name+'.txt', samples_labels)

    print n, 'images saved'
    
if __name__ == '__main__':

    print '... is generating samples'
    generate_img('mnist.pkl.gz', n=5)
    print 'done'
    
