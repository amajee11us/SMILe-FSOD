#!/bin/bash

export PATH="/home/ubuntu/anaconda/bin:$PATH"

# install pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# Install barebone dependencies
pip install cython 
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install yacs
pip install iopath
pip install pillow==9.5.0
pip install albumentations
