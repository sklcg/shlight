# SHLight
These are codes of the paper Learning Scene Illumination by Pairwise Photos from Front and Rear Mobile Cameras. For more details about our work (paper, slide, demo and dataset), please see the [project page](http://cg.tuoo.me/project/shlight).

## Dependencies
  + Python >= 3.6.0
  + Opencv >=  3.4.0
  + Tensorflow >= 1.7.0

## Usage
  1. Download the data, models and weights from [here](https://drive.google.com/drive/folders/16RH9lnAga4wdWQUuTUIjL8I7yoR-WWMe)(Some files need to be decompressed).
  2. Place these files to the correct location.
  3. cd to the path: src/, and run following commands.
    
 #### Train
    python3 train.py {Network} {Method} {Fusion} {Weight} [GPU]
  we supply some pretrained models. If you want to use these pretrained models, download them and run predict.py or evaluate.py directly. If you want retrain these models, please modify the value of **model_root** in [common.py](./src/common.py) before training, otherwise the program will continue training from our pretrained models.
  
 #### Evaluate
    python3 evaluate.py {Network} {Method} {Fusion} {Weight} [GPU]

 #### Predict
    python3 predict.py {Network} {Method} {Fusion} {Weight} [GPU]
 It will predict SH coefficents from pairwise images according to the [data/list/pred.txt](./data/list/pred.txt). You can predict the SH coefficients from other images by modifing this file (or modify the function *gen_pred_txt()* in [predict.py](./src/predict.py)).
___
##### *Arguments*
  + **Network** - ***[alexnet | vgg16 | googlenet | resnet]***
    + the feature extraction network
  + **Method** - ***[freeze | fromscratch | finetune]***
    + the method of treating the feature extraction network.
  + **Fusion** - ***[concat | subtract]***
    + the method of fusing the features
  + **Weight** - ***float value***
    + the weight of the SH loss, and 1 - Weight is the weight of the render loss.
  + ***GPU*** - ***optional, string of digitals***
    + specify the gpu device(s) for running. default: 0. 
    
##### *for example*
    python3 train.py alexnet freeze concat 0.8
    
  Then it will use alexnet as the backbone network, freeze the pretrained weights during traning, fuse extracted features by concatenation, and use **0.8\*SHLoss + 0.2\*RenderLoss** as the loss function. Since GPU device was not specified, it will be default running on gpu device: 0.

  This is also the best model in our paper (alexnet, freeze, concat, 0.8), you can use this model to compare with other works.


## Trouble Shooting
