import tensorflow as tf
import os 

epochs = 100
batch_size = 64

data_root = '../data/'
model_root = '../models/'

with open(os.path.join(data_root, 'list', 'train.txt')) as f:
    training_examples = len(f.readlines())

with open(os.path.join(data_root, 'list', 'test.txt')) as f:
    testing_examples = len(f.readlines())

summary_flag = True

def shrender(sh, shmap):
    # sh    : tensor [batchsize, 48]
    # shmap : tensor [height, width, 16]
    shmap = tf.expand_dims(shmap,  0) #[1, height, width, 16]
    shmap = tf.expand_dims(shmap, -1) #[1, height, width, 16, 1]
    shmap = tf.concat([shmap, shmap, shmap], axis=-1) #[1, height, width, 16, 3]

    sh = tf.reshape(sh, [-1, 1, 1, 16, 3]) #[batchsize, 1, 1, 16, 3]
    result = tf.reduce_sum(shmap * sh, axis = 3) #[batchsize, height, width, 3]
    
    return result

def visualize(sh, shmap):
    result = shrender(sh, shmap)

    result = tf.clip_by_value(result, 0.0, 1.0)
    result = tf.pow(result, 1/2.2) # gamma 
    result = tf.cast(tf.multiply(result, 255.0), tf.uint8)
    
    return result