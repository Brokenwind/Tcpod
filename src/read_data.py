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
    imgs = np.reshape(imgs,[imgNum,1,width*height])
    print ("load imgs finished")
    return imgs

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

def loadAll():
    train_imgs = load_image("../dataset/MNIST/train-images-idx3-ubyte.gz")
    train_labels = load_label("../dataset/MNIST/train-labels-idx1-ubyte.gz")
    test_imgs = load_image("t10k-images-idx3-ubyte.gz")
    test_labels = load_label("t10k-labels-idx1-ubyte.gz")
