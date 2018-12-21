import tensorflow as tf
import os 

from .net_places365 import VGG_Places365
from .net_places365 import Res152_Places365
from .net_places365 import Alex_Places365
from .net_places365 import Googlenet_Places365

#default settings
padding = 'SAME' # padding for convolutional layers
drop_rate = 0.2  # drop rate for dense layers
activation = ['bn', 'leakyrelu'] # activation for conv. layers and dense layers, except the output.

weights_path = '../weights/'

class Network():    
    def __init__(self, is_training = True):
        self.is_training = is_training
        self.fnmap = {'bn':self.bn, 'relu': self.relu, 'leakyrelu':self.leaky_relu}

    def output(self):
        return self.value

    def feed(self, inputs):
        self.value = inputs
        return self

    def bn(self):
        self.value = tf.layers.batch_normalization(self.value)
        return self

    def relu(self):
        self.value = tf.nn.relu(self.value)
        return self

    def leaky_relu(self):
        self.value = tf.nn.leaky_relu(self.value) # alpha = 0.2
        return self

    def max_pooling(self, size, strides, padding = 'SAME'):
        self.value = tf.layers.max_pooling2d(self.value, (size, size), (strides, strides), padding=padding)
        return self

    def avg_pooling(self, size, strides, padding = 'SAME'):
        self.value = tf.layers.average_pooling2d(self.value, (size, size), (strides, strides), padding=padding)
        return self

    def conv(self, filters, strides = 1, size = 3, activation = activation):
        self.value = tf.layers.conv2d(self.value, filters, (size,size), (strides, strides), padding='SAME')
        for fn in activation:
            self.fnmap[fn]()
        return self

    def deconv(self, filters, strides = 1, size = 3, activation = activation):
        self.value = tf.layers.conv2d_transpose(self.value, filters, (size,size), (strides, strides), padding='SAME')
        for fn in activation:
            self.fnmap[fn]()
        self.conv(filters, 3)
        self.conv(filters, 3)
        return self

    def dense(self, unit, drop_rate = drop_rate, activation = activation):
        self.value = tf.layers.flatten(self.value)
        self.value = tf.layers.dense(self.value, unit)

        for fn in activation:
            self.fnmap[fn]()

        if self.is_training and drop_rate > 0.0:
            self.value = tf.layers.dropout(self.value, drop_rate)

        return self

    def denses(self, unit_seq, last_activation = False, drop_rate = drop_rate):
        for unit in unit_seq[:-1]:
            self.dense(unit, drop_rate = drop_rate)

        if last_activation:
            self.dense(unit_seq[-1])
        else:
            self.dense(unit_seq[-1], drop_rate = 0.0, activation = [])

        return self

    def vgg_places365(self, from_scratch = False, trainable = True):
        feature = tf.add(self.value, 1.0)
        feature = tf.multiply(feature, 128.0)
        feature = tf.reverse(feature, axis = [-1])

        if from_scratch:
            weight = None
        else:
            weight = os.path.join(weights_path,'vgg_places365.mat')
            
        vgg = VGG_Places365({"data":feature}, trainable = trainable, weight = weight)
        self.value = vgg.get_output()

        return self

    def res_places365(self, from_scratch = False, trainable = True):
        feature = tf.add(self.value, 1.0)
        feature = tf.multiply(feature, 0.5)
        feature = tf.reverse(feature, axis = [-1])

        if from_scratch:
            weight = None
        else:
            weight = os.path.join(weights_path,'res_places365.mat')
            
        res = Res152_Places365({"data":feature}, trainable = trainable, weight = weight)
        self.value = res.get_output()

        return self

    def alex_places365(self, from_scratch = False, trainable = True):
        feature = tf.add(self.value, 1.0)
        feature = tf.multiply(feature, 128.0)
        feature = tf.reverse(feature, axis = [-1])
        if from_scratch:
            weight = None
        else:
            weight = os.path.join(weights_path, 'alex_places365.mat')

        alex = Alex_Places365({"data":feature}, trainable = trainable, weight = weight)
        self.value = alex.get_output()

        return self

    def googlenet_places365(self, from_scratch = False, trainable = True):
        feature = tf.add(self.value, 1.0)
        feature = tf.multiply(feature, 128.0)
        feature = tf.reverse(feature, axis = [-1])
        if from_scratch:
            weight = None
        else:
            weight = os.path.join(weights_path,'googlenet_places365.mat')

        alex = Googlenet_Places365({"data":feature}, trainable = trainable, weight = weight)
        self.value = alex.get_output()

        return self

