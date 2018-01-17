import numpy as np
import struct
import gzip

def load_image(filename):
    print ("load image set",filename)
    binfile= gzip.open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII' , buffers ,0)
    print ("head,",head)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'

    imgs = struct.unpack_from(bitsString,buffers,offset)

    binfile.close()
    imgs = np.reshape(imgs,[imgNum,width*height])
    print ("load imgs finished")

    return imgs/225.

def load_label(filename):

    print ("load label set",filename)
    binfile = gzip.open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II' , buffers ,0)
    print ("head,",head)
    imgNum=head[1]
    offset = struct.calcsize('>II')
    numString = '>'+str(imgNum)+"B"
    labels = struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels = np.reshape(labels,[imgNum,1])

    print ('load label finished')
    return labels

def load_all():
    train_imgs = load_image("../dataset/MNIST/train-images-idx3-ubyte.gz")
    train_labels = load_label("../dataset/MNIST/train-labels-idx1-ubyte.gz")
    test_imgs = load_image("../dataset/MNIST/t10k-images-idx3-ubyte.gz")
    test_labels = load_label("../dataset/MNIST/t10k-labels-idx1-ubyte.gz")
    # expand the labels
    train_labels = expand(train_labels)
    test_labels = expand(test_labels)
    print(train_imgs.shape)
    print(train_labels.shape)
    return train_imgs, train_labels, test_imgs, test_labels

def expand(y,n=10):
    """expandY(y,n)
    use vector to express each y[i]

    y: the array you will expand
    n: the number of class
    """
    m = np.size(y)
    yset = np.eye(n)
    # the result matrix
    yres = np.zeros((m,n))
    for i in np.arange(0,m):
        yres[i] = yset[y[i]]
    return yres

def gen_batches(n, batch_size):
    """Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.
    Examples
    --------
    >>> from sklearn.utils import gen_batches
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

if __name__ == '__main__':
    load_all()
