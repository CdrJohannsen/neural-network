import numpy as np
import struct

def openData():
    with open('data/train-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        images = images.reshape((size, nrows, ncols))
    with open('data/train-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        labels = labels.reshape((size,))
    for i in range(len(labels)):
        yield (labels[i], images[i])
