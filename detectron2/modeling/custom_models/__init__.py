# Copyright (c) Facebook, Inc. and its affiliates.

from .deeplabV2 import res_deeplab, DeepLabV2Head
from .deeplabV2_bis import DeepLabV2_backbone, DeepLabV2_head

__all__ = [k for k in globals().keys() if not k.startswith("_")]
# TODO can expose more resnet blocks after careful consideration
