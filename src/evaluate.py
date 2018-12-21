import numpy as np
import cv2 as cv
import tensorflow as tf
import os,sys,time,shutil
from skimage.measure import compare_ssim

from common import data_root, model_root
from common import batch_size, epochs, training_examples, testing_examples

import data
import model

def gen_eval_txt():
    """
    You can modify codes below to evaluate other images,
    input images should be resized to 224x224

    Format of eval.txt:
        path_to_front_view  path_to_rear_view  path_to_ground_truth_rendering
        path_to_front_view  path_to_rear_view  path_to_ground_truth_rendering
        path_to_front_view  path_to_rear_view  path_to_ground_truth_rendering
        ....
    
    The evaluator will predict SH coefficients from the two input images.
        Then, the prediction will be rendered using the predicted SH coefficients and the shmap.
        Finally it will be compared with the ground truth rendering.

    You should make sure that the resolution of shmap and gt image should be the same.
    """
    with open(os.path.join(data_root, 'list', 'test.txt')) as f:
        content = f.readlines()

    with open(os.path.join(data_root, 'list', 'eval.txt'), 'w') as f:
        for line in content:
            f.write(line.replace('/sh/','/gt/').replace('.txt', '_horse.jpg'))

class Evaluator:
    def __init__(self):
        shmap = np.load(os.path.join(data_root, 'shmap', 'horse.npy'))
        self.shmap = shmap.reshape((360, 480, 16, 1))

        mask = np.abs(self.shmap).sum(axis=-2).reshape((360, 480, 1))
        self.mask = mask.astype(np.bool).astype(np.uint8)

        self.mask_count = np.sum(self.mask)

        self.rmse = 0.0
        self.ssim = 0.0
        self.count = 0

    def _get_image_gt(self, filename):
        image = cv.imread(filename)[:,:,(2,1,0)] #convert BGR to RGB
        return image / 255.0

    def _render(self, sh):
        sh = sh.reshape((1, 1, 16, 3))
        image = np.sum(sh * self.shmap, axis = -2)

        image = np.clip(image, 0, 1)
        image = image ** (1/2.2)

        return image

    def _visualize(self, gt, pred):
        pred = pred * self.mask + gt * (1 - self.mask)
        cv.imshow('gt', gt[:,:,(2,1,0)])
        cv.imshow('pd', pred[:,:,(2,1,0)])
        assert cv.waitKey(0) != 27

    def _compute_rmse(self, gt, pred):
        # RMSE on unmasked images
        # return np.sqrt(np.mean(np.square(gt - pred)))

        # RMSE on masked images
        channel_mean = np.mean(np.square(gt - pred), axis = -1)
        return np.sqrt(np.sum(channel_mean) / self.mask_count)

    def _compute_ssim(self, gt, pred):
        return compare_ssim(gt[30:330,40:410,:], pred[30:330,40:410,:], multichannel = True)

    def feed(self, sh, gt_filename):
        image_gt = self._get_image_gt(gt_filename)
        image_pred = self._render(sh)


        rmse = self._compute_rmse(image_gt * self.mask, image_pred)
        self.rmse += rmse

        ssim = self._compute_ssim(image_gt * self.mask, image_pred)
        self.ssim += ssim

        print(gt_filename)
        print('RMSE: %.4f, DSSIM: %.4f'%(rmse, 1 - ssim))

        self.count += 1

        # visualization
        # self._visualize(image_gt, image_pred)

    def get_metric(self):
        return {
            'rmse' : self.rmse / self.count,
            'dssim' : 1 - self.ssim / self.count,
        }


if __name__ == '__main__':
    gen_eval_txt()

    tf.logging.set_verbosity(tf.logging.INFO)
    run_dir = os.path.join(model_root, '_'.join(sys.argv[1:5]))

    params = {
        'network' : (sys.argv[1], sys.argv[2]),
        'fuse_method' : sys.argv[3],
        'loss_weight' : float(sys.argv[4]),
        'run_dir': run_dir,
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = "0" if len(sys.argv) < 6 else sys.argv[5]

    estimator = tf.estimator.Estimator(model.model_fn, run_dir, params = params)

    predictions = estimator.predict(data.eval_input)

    evaluator = Evaluator()

    for item in predictions:
        evaluator.feed(item['sh'], item['gt_file'].decode())
        metric = evaluator.get_metric()
        print('Mean RMSE: %.4f, Mean DSSIM: %.4f\n'%(metric['rmse'], metric['dssim']))

    metric = evaluator.get_metric()
    print('Mean RMSE: %.4f, Mean DSSIM: %.4f\n'%(metric['rmse'], metric['dssim']))
