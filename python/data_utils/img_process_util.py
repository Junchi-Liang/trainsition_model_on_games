from os import listdir
from os.path import isfile, join, isdir
import glob
import numpy as np
import numpy.random
import scipy.misc

def img_normalize(img, mean_img):
    """
        normalize image
        img : numpy.array
        img - image(s) to be preprocessed, it can be one image or a batch of images
        mean_img : numpy.array
        mean_img - mean image of the dataset
        return img_processed
        img_processed : numpy.array
        img_processed - result image, img_processed = (input image - mean image) / 255
    """
    if (mean_img is None):
        img_processed = img
    else:
        img_processed = (np.float32(img) - np.float32(mean_img)) / 255.0
    return img_processed
