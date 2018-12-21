import tensorflow as tf
import os,sys,time

import data 
import losses
import network

from common import summary_flag, visualize

NetClass = {
    # freeze weights
    ('vgg16','freeze'): network.FixedVGG,
    ('resnet','freeze'): network.FixedResnet,
    ('alexnet','freeze'): network.FixedAlexnet,
    ('googlenet','freeze'): network.FixedGooglenet,

    # fine tune from pretrained model
    ('vgg16','finetune'): network.FinetuneVGG,
    ('resnet','finetune'): network.FinetuneResnet,
    ('alexnet','finetune'): network.FinetuneAlexnet,
    ('googlenet','finetune'): network.FinetuneGooglenet,

    # train from scratch
    ('vgg16','fromscratch'): network.FromScratchVGG,
    ('resnet','fromscratch'): network.FromScratchResnet,
    ('alexnet','fromscratch'): network.FromScratchAlexnet,
    ('googlenet','fromscratch'): network.FromScratchGooglenet,
}

def selectSHMap(concated_shmap):
    index = tf.random_uniform([1], maxval = 10, dtype=tf.int32)
    offset = tf.multiply(256, tf.minimum(index, 3))
    begin = tf.concat([offset, tf.constant([0,0])], axis=-1)
    selector = tf.slice(concated_shmap, begin, [256,512,16])
    return selector

def summary(features, labels, pred, shmaps):
    if not summary_flag: return [], []

    loss_mse = losses.loss_mse(pred['sh'], labels['sh'])
    loss_1st = losses.loss_mse_1st(pred['sh'], labels['sh'])
    loss_2nd = losses.loss_mse_2nd(pred['sh'], labels['sh'])
    loss_3rd = losses.loss_mse_3rd(pred['sh'], labels['sh'])
    loss_env = losses.metric_render(pred['sh'], labels['sh'], shmaps['envmap'])

    # show render result of envmap
    env_gt = visualize(labels['sh'], shmaps['envmap'])
    env_pred = visualize(pred['sh'], shmaps['envmap'])

    # show render result of envmap / sphere / dragon / bunny
    # sm =  selectSHMap(shmaps['concated'])
    # env_gt = visualize(labels['sh'], sm)
    # env_pred = visualize(pred['sh'], sm)

    render_gt = visualize(labels['sh'], shmaps['bunny'])
    render_pred = visualize(pred['sh'], shmaps['bunny'])

    front = tf.image.resize_bilinear(features['front_view'],[256,256])
    rear = tf.image.resize_bilinear(features['rear_view'],[256,256])

    views = tf.concat([front,rear], axis = 1)
    views = tf.add(views, 1.0)
    views = tf.cast(tf.multiply(views, 128.0), tf.uint8)

    envs = tf.concat([env_pred, env_gt], axis = 1)
    renders = tf.concat([render_pred, render_gt], axis = 1)

    summary_image = [
        tf.summary.image('result', tf.concat([views, envs, renders], axis = 2), max_outputs = 8)
    ]

    summary_scalar = [
        tf.summary.scalar('loss_sh', loss_mse),
        tf.summary.scalar('loss_1st', loss_1st),
        tf.summary.scalar('loss_2nd', loss_2nd),
        tf.summary.scalar('loss_3rd', loss_3rd),
        tf.summary.scalar('loss_render', loss_env)]

    return summary_image, summary_scalar

def model_fn(features, labels, mode, params):
    shmaps = data.load_shmap()

    #this is a dynamic random selector, the shmap used for computing render loss is changing during training.
    shmap = selectSHMap(shmaps['concated']) 

    net = NetClass[params['network']](features, training = (mode == tf.estimator.ModeKeys.TRAIN), fuse_method = params['fuse_method'])
    pred = net.inference()

    if mode == tf.estimator.ModeKeys.PREDICT:
        features.update(pred)
        return tf.estimator.EstimatorSpec(mode = mode, predictions = features)
        
    loss = losses.loss(pred['sh'], labels['sh'], shmap, params['loss_weight'])

    summary_image, summary_scalar = summary(features, labels, pred, shmaps)
 
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()

        learning_rate = tf.train.exponential_decay(5e-4, global_step, 20000, 0.9, staircase = False)
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        hooks = [
            tf.train.SummarySaverHook(save_steps=50, output_dir=params['run_dir']+'/eval', summary_op = tf.summary.merge(summary_image))]
        metrics = {
            'loss_sh' : tf.metrics.mean(losses.loss_mse(pred['sh'], labels['sh'])),
            'loss_1st' : tf.metrics.mean(losses.loss_mse_1st(pred['sh'], labels['sh'])),
            'loss_2nd' : tf.metrics.mean(losses.loss_mse_2nd(pred['sh'], labels['sh'])),
            'loss_3rd' : tf.metrics.mean(losses.loss_mse_3rd(pred['sh'], labels['sh'])),
            'loss_render' : tf.metrics.mean(losses.metric_render(pred['sh'], labels['sh'], shmaps['envmap']))
        }
        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL, loss = loss, evaluation_hooks=hooks, eval_metric_ops = metrics)

    