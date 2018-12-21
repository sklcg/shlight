import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)

import os,sys,time

from common import model_root, batch_size, epochs, training_examples , testing_examples
import data
import model

tf.logging.set_verbosity(tf.logging.INFO)
run_dir = os.path.join(model_root, '_'.join(sys.argv[1:5]))

params = {
    'network' : (sys.argv[1], sys.argv[2]),
    'fuse_method' : sys.argv[3],
    'loss_weight' : float(sys.argv[4]),
    'run_dir': run_dir,
}

os.environ['CUDA_VISIBLE_DEVICES'] = "0" if len(sys.argv) < 6 else sys.argv[5]

estimator = tf.estimator.Estimator(model.model_fn, run_dir, params = params, config=tf.estimator.RunConfig(tf_random_seed=0))

for i in range(epochs):
    tf.set_random_seed(i + 1)

    print('training')
    estimator.train(data.train_input, steps = training_examples // batch_size)

    print('evaluation')
    estimator.evaluate(data.test_input, steps = testing_examples // batch_size)
