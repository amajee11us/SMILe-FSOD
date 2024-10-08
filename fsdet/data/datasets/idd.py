# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from fsdet.structures import BoxMode
from fsdet.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_idd_detection"]


# fmt: off
CLASS_NAMES = [
    "motorcycle", "rider", "person", "car",
    "autorickshaw", "truck", "bus", "bicycle",
    "traffic sign", "traffic light",
]
# fmt: on


def load_idd_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            if cls in CLASS_NAMES:
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                # Original annotations are integers in the range [1, W or H]
                # Assuming they mean 1-based pixel indices (inclusive),
                # a box with annotation (xmin=1, xmax=W) covers the whole image.
                # In coordinate space this is represented by (xmin=0, xmax=W)
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                instances.append(
                    {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_idd_detection(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_idd_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )
