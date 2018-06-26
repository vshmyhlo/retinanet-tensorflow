# Implementation of RetinaNet from [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) paper in [TensorFlow](https://www.tensorflow.org/)

## Differences from the Official Paper
For some reason this architecture is very hard to train, loss gets stuck at early stages of training, predicting everything as a background.
To overcome this problem I tried different initialization schemes, backbone architectures and losses. 
Also training on multiple GPUs might help.
You can choose densenet, resnext or mobilenet_v2 as a backbone architecture.


My current setup which gives me at least some result:
* Training on a single Titan X with 1 image per batch (cant fit into memory anything larger with 500 image scale)
* MobileNetV2 as a backbone, with 500 image scale (cant fit into memory anything larger)
* Not using Focal Loss, I am sure I will get back to it once I will find out why it is so hard to train it as it is described in a paper.
* Using combination of balanced cross-entropy and dice loss.

## Notes
Interestingly, other open source implementations I have found on github all using much lower learning rate (1e-4, 1e-5) and/or gradient clipping.