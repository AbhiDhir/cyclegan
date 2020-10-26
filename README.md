<!-- <img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>
-->

# CycleGAN (fork of [@xhujoy/CycleGAN-tensorflow](https://github.com/xhujoy/CycleGAN-tensorflow))

Tensorflow implementation for learning an image-to-image translation **without** input-output pairs.
The method is proposed by [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/) in
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkssee](https://arxiv.org/pdf/1703.10593.pdf).
For example in paper:

<img src="imgs/teaser.jpg" width="1000px"/>

## Getting Started

### Train

Dataset organization:

```
<dataset-name>/
    trainA/
        0001.png
        0002.png
        ...
    trainB/
        0001.png
        0002.png
        ...
    testA/
        0001.png
        0002.png
        ...
    testB/
        0001.png
        0002.png
        ...
```

Train a model:

```bash
python main.py --dataset_dir=/path/to/data
```

Use tensorboard to visualize the training details:

```bash
tensorboard --logdir=./logs
```

### Test

```bash
python main.py --dataset_dir=/path/to/data --phase=test --which_direction=AtoB
```
