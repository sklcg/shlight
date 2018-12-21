### Models

These are some trained models using our codes. You can directly use it to evaluate.

Each folder name represents the network architecture and training details.

    Network_TrainingMethod_FuseMethod_LossWeight

For using this model to test or evaluate, you can easily use this command.

    python3 evalute.py Network TrainingMethod FuseMethod LossWeight [GPU]

These models that are all trained  with the same configuration on the same training data.

If you train some models, the training process can be visualized through tensorboard.
    
    tensorboard --logdir=./models --port=6006

Using the same random seed (do not modify the random seed in [train.py](../src/train.py)), **similar**<sup>1</sup> result could be reproduced.
____
**1.** Even if we set the same random seed in tensorflow, it's still hardly to produce the same result for different runnings. Because some random behaviours do not rely on the random seed. They may produce extremely different result in different running process on different platforms/systems/machines, especially for parallel programs. See [this link](https://stackoverflow.com/q/42156296/9827138) for more details.