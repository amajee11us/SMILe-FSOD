#!/bin/bash

# Root directory for the dataset
DATA_ROOT="<Add your data-root here>"

# Directory where the VOC2007 dataset will be stored
DATASET_DIR="$DATA_ROOT/PASCAL_VOC"

# Download and prepare the PASCAL VOC 2007 dataset
if [ ! -d "$DATASET_DIR/VOCdevkit/VOC2007" ]; then
    echo "Downloading PASCAL VOC 2007 dataset..."
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

    # Extract the dataset
    echo "Extracting dataset..."
    mkdir -p $DATASET_DIR
    tar -xf VOCtrainval_06-Nov-2007.tar -C $DATASET_DIR
    tar -xf VOCtest_06-Nov-2007.tar -C $DATASET_DIR

    # Clean up the tar files
    rm VOCtrainval_06-Nov-2007.tar
    rm VOCtest_06-Nov-2007.tar

    echo "PASCAL VOC 2007 dataset is ready."
else
    echo "PASCAL VOC 2007 dataset already exists."
fi

# Download and prepare the PASCAL VOC 2012 dataset
if [ ! -d "$DATASET_DIR/VOCdevkit/VOC2012" ]; then
    echo "Downloading PASCAL VOC 2012 dataset..."
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

    # Extract the dataset
    echo "Extracting VOC 2012 dataset..."
    mkdir -p $DATASET_DIR
    tar -xf VOCtrainval_11-May-2012.tar -C $DATASET_DIR

    # Clean up the tar file
    rm VOCtrainval_11-May-2012.tar

    echo "PASCAL VOC 2012 dataset is ready."
else
    echo "PASCAL VOC 2012 dataset already exists."
fi

# Download the VOC few-shot splits
wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" http://dl.yf.io/fs-det/datasets/vocsplit/

# Link both VOC 2007 and 2012 datasets
ln -s $DATA_ROOT/PASCAL_VOC/VOCdevkit/VOC2007/ datasets/VOC2007
ln -s $DATA_ROOT/PASCAL_VOC/VOCdevkit/VOC2012/ datasets/VOC2012
