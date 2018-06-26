import sys

sys.path.append('../')

from loadminist import load

# load data
train_images, train_lables, test_images, test_lables = load('../minist').load_minist()
print('train_images shape:%s' % train_images.shape)
print('train_lables shape:%s' % train_lables.shape)
print('test_images shape:%s' % test_images.shape)
print('test_lables shape:%s' % test_lables.shape)

