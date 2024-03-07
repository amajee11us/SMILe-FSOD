#!/bin/bash

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install barebone dependencies
pip install cython 
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install yacs
pip install iopath
pip install pillow==9.5.0
pip install albumentations
