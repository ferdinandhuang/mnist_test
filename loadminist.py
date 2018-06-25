import numpy as np
import os
import struct

class load():
    def load_mnist(path = 'minist'):
        """Read train and test dataset and lables from path"""

        train_image_path = 'train-images.idx3-ubyte'
        train_lable_path = 'train-labels.idx1-ubyte'

        test_image_path = 't10k-images.idx3-ubyte'
        test_lable_path = 't10k-labels.idx1-ubyte'

        with open(os.path.join(path, train_lable_path), 'rb') as labelpath:
            magic, n = struct.unpack('>II',
                                     labelpath.read(8))
            labels = np.fromfile(labelpath,
                                 dtype=np.uint8)
            train_lables = labels

        with open(os.path.join(path, train_image_path), 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                                   imgpath.read(16))
            images = np.fromfile(imgpath,
                                 dtype=np.uint8)
            train_images = images

        with open(os.path.join(path, test_lable_path), 'rb') as labelpath:
            magic, n = struct.unpack('>II',
                                     labelpath.read(8))
            labels = np.fromfile(labelpath,
                                 dtype=np.uint8)
            test_lables = labels

        with open(os.path.join(path, test_image_path), 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                                   imgpath.read(16))
            images = np.fromfile(imgpath,
                                 dtype=np.uint8)
            test_images = images

        return train_images,train_lables,test_images,test_lables


