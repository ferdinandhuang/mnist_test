import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class load:
    def __init__(self,
                 path='mnist'):
        self.path = path
        self.batch_num = 0
        train_images, train_labels, test_images, test_labels = self._load_mnist()

        # One Hot Encode
        ohe = OneHotEncoder()
        ohe.fit([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
        train_labels_onehot = ohe.transform(train_labels).toarray()
        test_labels_onehot = ohe.transform(test_labels).toarray()

        self.train_images = train_images/255
        self.train_labels_onehot = train_labels_onehot
        self.train_labels = train_labels
        self.test_images = test_images/255
        self.test_labels_onehot = test_labels_onehot
        self.test_labels = test_labels

    def _load_mnist(self):
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

    def get_all_data_onehot(self):
        return self.train_images, self.train_labels_onehot, self.test_images, self.test_labels_onehot

    def get_all_data(self):
        return self.train_images, self.train_labels, self.test_images, self.test_labels

    def get_next_batch_onehot(self, batch_size=64, type="train"):
        """
        get data by batch_size
        :param batch_size: number of batch
        :return: batch of data
        """

        train_images= self.train_images
        test_images= self.test_images

        if type == "train":
            start = self.batch_num
            if start == train_images.shape[0]:
                self.batch_num = 0
                start = 0

            train_labels_onehot = self.train_labels_onehot

            if start + batch_size <= train_images.shape[0]:
                end = start + batch_size
                self.batch_num = self.batch_num + batch_size
            else:
                end = train_images.shape[0]
                self.batch_num = end

            return train_images[start: end], train_labels_onehot[start: end]
        elif type == "test":
            test_labels_onehot = self.test_labels_onehot
            return test_images, test_labels_onehot
        else:
            raise Exception("Not validity type!")


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load()._load_mnist()
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
