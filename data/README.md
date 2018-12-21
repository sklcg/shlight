# Data 
This doc introduce how we generate and use the training data. 
Ignoring these descriptions, you can also run our code.
### Folders
#### > im
These images are extracted from HDR envmaps.

    rear_view  : 'im/%04d/%03d_0.jpg'%(envmap_id, sample_id)
    front_view : 'im/%04d/%03d_1.jpg'%(envmap_id, sample_id)
#### > sh
SH coefficients are computed using [Github:google/spherical harmonics](https://github.com/google/spherical-harmonics)

    sh_coefficient: 'sh/%04d/%03d.txt'%(envmap_id, sample_id)

#### > gt
Ground truth images are rendered from original envmaps by a physics-based rendering tool.

    ground truth rendering : 'gt/%04d/%03d.jpg'(envmap_id, sample_id)

#### > shmap
These are the pre-comuted SH coefficients map for computing the render loss or efficient rendering. You can load these shmaps by this code:

    shmap = np.load('item.npy').reshape(height, width, 16, 1)

#### > list
These are some text files that indicate which data is to be used for training, testing, evaluation and prediction.

### Generate
These data are extracted from the HDR envmaps. We use 400 HDR envmaps ([download link](https://drive.google.com/drive/folders/1vY5qOvFDS3Elo-xoDuZTwiFGxR88Zq0e?usp=sharing)) for our training and testing. 
These envmaps are selected from our HDR dataset. This dataset could be found at our [project page](http://cg.tuoo.me/project/shlight).
For each sample, we randomly selected a view direction, compute the SH coefficient, and extract two views from the HDR envmap by this direction and its opposite direction.

### Filter
Using [select.py](./select.py) we can filter out the images over/under exposure to generate the [train.txt](./list/train.txt) and [test.txt](./list/test.txt) for training and testing.

    python select.py

Each line of the train.txt and test.txt contains three file paths.  

    path_to_front_view_file  path_to_rear_view_file  path_to_sh_file