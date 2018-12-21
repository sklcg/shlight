# Pretrained Weights of Places365
These mat files are converted from [Places365](https://github.com/CSAILVision/places365) by a [converting tool](https://github.com/dhaase-de/caffe-tensorflow-python3) that can convert caffee graph and weight to tensorflow code and weight. 

We download the prototxt and weights from [Places365](https://github.com/CSAILVision/places365) and convert it to the code and mat

**For example, after downloading the alexnet_places365.prototxt and alexnet_places365.caffemodel, we can get the weights and tensorflow code by this code**
    
    python convert.py --caffemodel alexnet_places365.caffemodel \
                        deploy_alexnet_places365.prototxt \
                      --data-output-path alexnet_places365.mat \
                      --code-output-path ./alexnet.py 

Then we merge the alexnet.py to our code ([networks](../src/networks)), and load the weights from the mat file

    np.load('alexnet_places365.mat').tolist()

