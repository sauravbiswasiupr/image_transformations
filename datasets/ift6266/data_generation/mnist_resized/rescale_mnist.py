import numpy,cPickle,gzip,Image,pdb,sys


def zeropad(vect,img_size=(28,28),out_size=(32,32)):
    delta = (numpy.abs(img_size[0]-out_size[0])/2,numpy.abs(img_size[1]-out_size[1])/2)
    newvect = numpy.zeros(out_size)
    newvect[delta[0]:-delta[0],delta[1]:-delta[1]] = vect.reshape(img_size)
    return newvect.flatten()

def rescale(vect,img_size=(28,28),out_size=(32,32), filter=Image.NEAREST):
    im = Image.fromarray(numpy.asarray(vect.reshape(img_size)*255.,dtype='uint8'))
    return (numpy.asarray(im.resize(out_size,filter),dtype='float32')/255.).flatten()

 
#pdb.set_trace()
def rescale_mnist(newsize=(32,32),output_file='mnist_rescaled_32_32.pkl',mnist=cPickle.load(gzip.open('mnist.pkl.gz'))):
    newmnist = []
    for set in mnist:
        newset=numpy.zeros((len(set[0]),newsize[0]*newsize[1]))
        for i in xrange(len(set[0])):
            print i,
            sys.stdout.flush()
            newset[i] = rescale(set[0][i])
        newmnist.append((newset,set[1]))
    cPickle.dump(newmnist,open(output_file,'w'),protocol=-1)
    print 'Done rescaling'


def zeropad_mnist(newsize=(32,32),output_file='mnist_zeropadded_32_32.pkl',mnist=cPickle.load(gzip.open('mnist.pkl.gz'))):
    newmnist = []
    for set in mnist:
        newset=numpy.zeros((len(set[0]),newsize[0]*newsize[1]))
        for i in xrange(len(set[0])):
            print i,
            sys.stdout.flush()
            newset[i] = zeropad(set[0][i])
        newmnist.append((newset,set[1]))
    cPickle.dump(newmnist,open(output_file,'w'),protocol=-1)
    print 'Done padding'

if __name__ =='__main__':
    print 'Creating resized datasets'
    mnist_ds = cPickle.load(gzip.open('mnist.pkl.gz'))
    #zeropad_mnist(mnist=mnist_ds)
    rescale_mnist(mnist=mnist_ds)
    print 'Finished.'
