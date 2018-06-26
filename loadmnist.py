import numpy as np
import os
import struct
import matplotlib.pyplot as plt


class load:
    def __init__(self,
                 path='mnist'):
        self.path = path

    def load_mnist(self):
        """Read train and test dataset and labels from path"""

        train_image_path = 'train-images.idx3-ubyte'
        train_label_path = 'train-labels.idx1-ubyte'

        test_image_path = 't10k-images.idx3-ubyte'
        test_label_path = 't10k-labels.idx1-ubyte'

        with open(os.path.join(self.path, train_label_path), 'rb') as labelpath:
            magic, n = struct.unpack('>II', labelpath.read(8))
            labels = np.fromfile(labelpath, dtype=np.uint8)
            train_labels = labels.reshape(len(labels), 1)

        with open(os.path.join(self.path, train_image_path), 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath,
                                 dtype=np.uint8).reshape(len(train_labels), 784)
            train_images = images

        with open(os.path.join(self.path, test_label_path), 'rb') as labelpath:
            magic, n = struct.unpack('>II', labelpath.read(8))
            labels = np.fromfile(labelpath,
                                 dtype=np.uint8)
            test_labels = labels.reshape(len(labels), 1)

        with open(os.path.join(self.path, test_image_path), 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)
            test_images = images

        return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load().load_mnist()
    print('train_images shape:%s' % str(train_images.shape))
    print('train_labels shape:%s' % str(train_labels.shape))
    print('test_images shape:%s' % str(test_images.shape))
    print('test_labels shape:%s' % str(test_labels.shape))

    np.random.seed(1024)

    trainImage = np.random.randint(60000, size=4)
    testImage = np.random.randint(10000, size=2)

    img1 = train_images[trainImage[0]].reshape(28, 28)
    label1 = train_labels[trainImage[0]]
    img2 = train_images[trainImage[1]].reshape(28, 28)
    label2 = train_labels[trainImage[1]]
    img3 = train_images[trainImage[2]].reshape(28, 28)
    label3 = train_labels[trainImage[2]]
    img4 = train_images[trainImage[3]].reshape(28, 28)
    label4 = train_labels[trainImage[3]]

    img5 = test_images[testImage[0]].reshape(28, 28)
    label5 = test_labels[testImage[0]]
    img6 = test_images[testImage[1]].reshape(28, 28)
    label6 = test_labels[testImage[1]]


    plt.figure(num='mnist', figsize=(2, 3))

    plt.subplot(2, 3, 1)
    plt.title(label1)
    plt.imshow(img1)

    plt.subplot(2, 3, 2)
    plt.title(label2)
    plt.imshow(img2)

    plt.subplot(2, 3, 3)
    plt.title(label3)
    plt.imshow(img3)

    plt.subplot(2, 3, 4)
    plt.title(label4)
    plt.imshow(img4)

    plt.subplot(2, 3, 5)
    plt.title(label5)
    plt.imshow(img5)

    plt.subplot(2, 3, 6)
    plt.title(label6)
    plt.imshow(img6)
    plt.show()

