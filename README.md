# SMILe: Leveraging Submodular Mutual Information For Robust Few-Shot Object Detection

This repo contains the implementation of proposed SMILe framework which introduces a combinatorial viewpoint in Few-Shot Object Detection. SMILe is built upon the codebase [FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags).

![SMILe Figure](demo/images/overview_smile.png)

Confusion and forgetting of object classes have been challenges of prime interest in Few-Shot Object Detection (FSOD).
To overcome these pitfalls in metric learning based FSOD techniques, we introduce a novel Submodular Mutual Information Learning (**SMILe**) framework which adopts combinatorial mutual information functions to enforce the creation of tighter and discriminative feature clusters in FSOD.
Our proposed approach generalizes to several existing approaches in FSOD, agnostic of the backbone architecture demonstrating elevated performance gains.
A paradigm shift from instance based objective functions to combinatorial objectives in SMILe naturally preserves the diversity within an object class resulting in reduced forgetting when subjected to few training examples.
Furthermore, the application of mutual information between the already learnt (base) and newly added (novel) objects ensures sufficient separation between base and novel classes, minimizing the effect of class confusion.
Experiments on popular FSOD benchmarks, PASCAL-VOC and MS-COCO show that our approach generalizes to State-of-the-Art (SoTA) approaches improving their novel class performance by up to 5.7\% (3.3 $mAP$ points) and 5.4\% (2.6 $mAP$ points) on the 10-shot setting of VOC (split 3) and 30-shot setting of COCO datasets respectively. 
Our experiments also demonstrate better retention of base class performance and up to $2\times$ faster convergence over existing approaches agnostic of the underlying architecture.

## Installation
The installation instructions are similar to FsDet and FSCE.
FsDet is built on [Detectron2](https://github.com/facebookresearch/detectron2). But you don't need to build detectron2 seperately as this codebase is self-contained. You can follow the instructions below to install the dependencies and build `FsDet`. FSCE functionalities are implemented as `class`and `.py` scripts in FsDet which therefore requires no extra build efforts. 

**Dependencies**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3 
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* Dependencies: ```pip install -r requirements.txt```
* pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
* [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'``` 
* [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
* GCC >= 4.9

**Build**

```bash
python setup.py build develop  # you might need sudo
```
Note: you may need to rebuild FsDet after reinstalling a different build of PyTorch.



## Data preparation

We adopt the same benchmarks as in FsDet, including three datasets: PASCAL VOC, COCO and LVIS. 

- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [fsdet/data/datasets/builtin_meta.py](fsdet/data/datasets/builtin_meta.py).
- [COCO](http://cocodataset.org/): We use COCO 2014 without COCO minival for training and the 5,000 images in COCO minival for testing. We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.
- [LVIS](https://www.lvisdataset.org/): We treat the frequent and common classes as the base classes and the rare categories as the novel classes.

The datasets and data splits are built-in, simply make sure the directory structure agrees with [datasets/README.md](datasets/README.md) to launch the program. 

The default seed that is used to report performace in research papers can be found [here](http://dl.yf.io/fs-det/datasets/).

## Train & Inference

### Training

We follow the eaact training procedure of FsDet and we use **random initialization** for novel weights. For a full description of training procedure, see [here](https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/TRAIN_INST.md).

#### 1. Stage 1: Training base detector.

```
python tools/train_net.py --num-gpus 4 \
        --config-file configs/PASCAL_VOC/base-training/R101_FPN_base_training_split1.yml
```

#### 2. Random initialize  weights for novel classes.

```
python tools/ckpt_surgery.py \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

This step will create a `model_surgery.pth` from` model_final.pth`. 

Don't forget the `--coco` and `--lvis`options when work on the COCO and LVIS datasets, see `ckpt_surgery.py` for all arguments details.

#### 3. Stage 2: Few-Shot Adaptation on novel data.

```
python tools/train_net.py --num-gpus 4 \
        --config-file configs/PASCAL_VOC/split1/split1_10shot_FSCE_FLQMI_IoU_0.7_weight_0.5.yaml \
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```

Where `WEIGHTS_PATH` points to the `model_surgery.pth` generated from the previous step. Or you can specify it in the configuration yml. 

#### Evaluation

To evaluate the trained models, run

```angular2html
python tools/test_net.py --num-gpus 8 \
        --config-file configs/PASCAL_VOC/split1/split1_10shot_FSCE_FLQMI_IoU_0.7_weight_0.5.yaml \
        --eval-only
```

Or you can specify `TEST.EVAL_PERIOD` in the configuation yml to evaluate during training. 



### Multiple Runs

For ease of training and evaluation over multiple runs, fsdet provided several helpful scripts in `tools/`.

You can use `tools/run_experiments.py` to do the training and evaluation. For example, to experiment on 30 seeds of the first split of PascalVOC on all shots, run

```angular2html
python tools/run_experiments.py --num-gpus 4 \
        --shots 1 5 10 --seeds 0 10 --split 1
```

### Acknowledgement
We thank the authors of the below mentioned contributions. 
Most of our code is adapted from the FSCE approach (CVPR 2021).

Frustratingly Simple Few-Shot Object Detection ([FsDet v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags))

Few-Shot Object Detection via Contrastive Proposal Encoding ([FSCE](https://github.com/megvii-research/FSCE))

Attention Guided Cosine Margin For Overcoming Class-Imbalance in Few-Shot Road Object Detection ([AGCM](https://arxiv.org/abs/2111.06639))



