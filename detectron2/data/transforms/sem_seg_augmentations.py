import numpy as np
import torch
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform

class CutMix(T.Augmentation):
    def get_transform(self, image1, image2, label1, label2):
    	pass

class CutOutPolicy(T.Augmentation):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def get_transform(self, image):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = image.shape[0]
        w = image.shape[1]

        mask = np.ones(image.shape, np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        return CutOut(mask)

class CutOut(Transform):
    def __init__(self, mask):
        self.mask = mask        

    def apply_image(self, img, seg_mode=False):
        if seg_mode:
            img = img * self.mask[:,:,0]
            img = img + ((1-self.mask[:,:,0])*19) # void class 19 on Cityscapes train
        else:
            img = img * self.mask
        return img

    def apply_segmentation(self, segmentation):
        return self.apply_image(segmentation, seg_mode=True)

    def apply_coords(self, coords):
        return coords