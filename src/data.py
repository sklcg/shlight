import tensorflow as tf
import sys, os, random
import numpy as np

from common import batch_size, data_root

def load_shmap():
    def _load(filename, height, width):
        filename = os.path.join(data_root, 'shmap', filename)
        return tf.constant(np.load(filename).reshape((height, width, 16)), tf.float32)

    shmap = {'sphere':(256,256), 'bunny':(256,256), 'dragon':(256,256), 
              'envmap':(256,512), 'concated':(1024,512)}

    for item in shmap:
        shmap[item] = _load(item + '.npy', *shmap[item])

    return shmap

def load_rgb(filename):
    raw = tf.read_file(filename)
    image = tf.image.decode_image(raw)
    image = tf.reshape(image,[224,224,3])
    image = tf.multiply(tf.cast(image,tf.float32), 1/128.0)
    image = tf.subtract(image, 1.0)
    return image

def load_sh(filename):
    raw = tf.read_file(filename)
    part = tf.string_split([raw],'\r\n ').values
    return tf.string_to_number(part)

def load_data(front_file, rear_file, sh_file):
    front = load_rgb(front_file)
    rear = load_rgb(rear_file)
    sh = load_sh(sh_file)
    return {"front_view" : front, "rear_view" : rear}, { "sh" : sh}

def decode_filename(textline):
    front_file = tf.string_split([textline], ' ').values[0]
    rear_file = tf.string_split([textline], ' ').values[1]
    sh_file = tf.string_split([textline], ' ').values[2]
    return front_file, rear_file, sh_file

def input_fn(filename, is_training = True):
    data = tf.data.TextLineDataset(filename).map(decode_filename).map(load_data)
    if is_training:
        data = data.shuffle(buffer_size = batch_size * 2).repeat()
    return data.batch(batch_size)

def train_input():
    return input_fn(os.path.join(data_root, 'list', 'train.txt'), is_training = True)

def test_input():
    return input_fn(os.path.join(data_root, 'list', 'test.txt'), is_training = False)

def eval_input():
    def _load_data(front_file, rear_file, gt_file):
        return {"front_view" : load_rgb(front_file), "rear_view" : load_rgb(rear_file), "gt_file" : gt_file}, {}
    data = tf.data.TextLineDataset(os.path.join(data_root, 'list', 'eval.txt')).map(decode_filename).map(_load_data)
    return data.batch(batch_size)

def pred_input():
    def _load_data(front_file, rear_file, sh_file):
        return {"front_view" : load_rgb(front_file), "rear_view" : load_rgb(rear_file), "save_path" : sh_file}, {}
    data = tf.data.TextLineDataset(os.path.join(data_root, 'list', 'pred.txt')).map(decode_filename).map(_load_data)
    return data.batch(batch_size)

