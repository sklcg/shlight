import os,sys,time
import tensorflow as tf
from common import data_root, model_root, batch_size, epochs, training_examples , testing_examples
import data
import model

def gen_pred_txt(tag):
    with open(os.path.join(data_root, 'list', 'pred.txt'),'w') as file:
        """
        You can modify codes below to predict SH coefficients from other images,
        input images should be resized to 224x224

        Format of pred.txt:
            path_to_front_view  path_to_rear_view  path_to_save_predicted_SH_coefficients
            path_to_front_view  path_to_rear_view  path_to_save_predicted_SH_coefficients
            path_to_front_view  path_to_rear_view  path_to_save_predicted_SH_coefficients
            ....

        """
        for i in range(1, 32):
            for j in range(512):
                file.write(os.path.join(data_root, 'im/%04d/%03d_1.jpg '%(i,j)))
                file.write(os.path.join(data_root, 'im/%04d/%03d_0.jpg '%(i,j)))
                file.write(os.path.join(data_root, 'prediction/%s/%04d/%03d.txt\n'%(tag, i,j)))

def sh_write(filename, shlist):
    print('saving %s'%filename)

    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename,'w') as sh:
        for i in range(16):
            for j in range(3):
                sh.write('%f '%shlist[i*3+j])
            sh.write('\n')

if __name__ == '__main__':
    gen_pred_txt('_'.join(sys.argv[1:5]))

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

    predictions = estimator.predict(data.pred_input)
    for pred in predictions:
        sh_write(pred['save_path'].decode(), pred['sh'].tolist())
