import tensorflow as tf
from common import shrender

def loss_render(sh_pred, sh_gt, shmap):
    pred = shrender(sh_pred, shmap)
    gt = shrender(sh_gt, shmap)
    return tf.reduce_mean(
        tf.reduce_sum(tf.square(pred - gt), axis=[1,2]) / tf.cast(tf.count_nonzero(tf.reduce_sum(shmap,[-1])), tf.float32)
    )

def loss_mse_weighted(pred, gt):
    weights = tf.constant([1/12]*3 + [1/36]*9 + [1/60]*15 + [1/84]*21)
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(pred - gt), weights), axis = -1))

def loss_mse_1st(pred, gt):
    weights = tf.constant([1 /3] * 3 + [0] * 45)
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(pred - gt), weights), axis = -1))

def loss_mse_2nd(pred, gt):
    weights = tf.constant([1/12] * 12 + [0] * 36)
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(pred - gt), weights), axis = -1))

def loss_mse_3rd(pred, gt):
    weights = tf.constant([1/27] * 27 + [0] * 21)
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(pred - gt), weights), axis=-1))

def loss_mse_4th(pred, gt):
    weights = tf.constant([1/48] * 48)
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.square(pred - gt), weights), axis=-1))

def loss_mse(pred, gt):
    return tf.reduce_mean(tf.square(pred - gt))

def loss(pred, gt, shmap, ratio):
    shloss = loss_mse_weighted(pred, gt)
    rdloss = loss_render(pred, gt, shmap)
    return tf.multiply(shloss, ratio) + tf.multiply(rdloss, 1 - ratio)

def metric_render(sh_pred, sh_gt, shmap):

    pred = tf.clip_by_value(shrender(sh_pred, shmap), 0.0, 1.0)
    gt = tf.clip_by_value(shrender(sh_gt, shmap), 0.0, 1.0)

    return tf.reduce_mean(tf.square(pred - gt))