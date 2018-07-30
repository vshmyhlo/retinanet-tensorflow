# Implementation of RetinaNet from [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) paper in [TensorFlow](https://www.tensorflow.org/)

### Differences from the Official Paper
For some reason this architecture is extremely hard to train, loss gets
stuck at early stages of training, predicting everything as a background 
(probably to the fact that i am using low batch size). To overcome this 
problem I tried different initialization schemes, backbone architectures 
and losses. 

You can choose densenet, resnext or mobilenet_v2 as a backbone architecture.

### Observations
* Training focal loss on a simple synthetic dataset does work but poorly 
* To overcome small batch size problem one might try training on multiple 
GPUs or using Group Normalization.
* Probably the most important observation is that using Group Normalization
instead of Batch Normalization gives significantly better results
(when trained on small batches)

### Current setup which gives at least some result:
* Training on a single Titan X with 1 image per batch
(cant fit into memory anything larger with 500 image scale)
* MobileNetV2 as a backbone, with 500 image scale 
(cant fit into memory anything larger)
* Not using Focal Loss, I am sure I will get back to it once I will
find out why it is so hard to train it as it is described in a paper.
* Using combination of balanced cross-entropy and dice loss.
* Using Group Normalization.

### Notes
Interestingly, other open source implementations I have found on github
all using much lower learning rate (1e-4, 1e-5) and/or gradient clipping.
