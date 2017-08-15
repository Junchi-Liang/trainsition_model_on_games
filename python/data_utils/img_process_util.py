from os import listdir
from os.path import isfile, join, isdir
import glob
import numpy as np
import numpy.random
import scipy.misc
import matplotlib.pyplot as plt

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

def dense_optical_flow_to_quivers(flow):
    """
        flow : numpy.ndarray
        flow - dense optical flow computed from opencv (e.g. optical flow from cv2.calcOpticalFlowFarneback)
        return [X, Y, U, V]
        parameters for matplotlib.pyplot.quiver
        X : 2d array
        X - The x coordinates of the arrow locations
        Y : 2d array
        Y - The y coordinates of the arrow locations
        U : 2d array
        U - The x components of the arrow vectors
        V : 2d array
        V - The y components of the arrow vectors
    """
    X, Y = np.meshgrid(np.arange(0.5, flow.shape[0], 1), np.arange(0.5, flow.shape[1], 1))
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[i, j] = flow[U.shape[0] - 1 - i, j, 0]
            V[i, j] = -flow[U.shape[0] - 1 - i, j, 1]
    return [X, Y, U, V]

def visualize_segmentation(self, seg_output, color):
    """
        construct a color image according to segmentation result
        seg_output : np.ndarray
        seg_output = segmentation result, shape (image height, image width)
        color : np.ndarray
        color : color mapping, shape (class, 3), the RGB for the i-th class is color[class]
        ------------------------------------------------------------------------------------------
        return image
        image : np.ndarray
        image = color image as segmentation output, shape (image height, image width, 3)
    """
    image = np.zeros([seg_output.shape[0], seg_output.shape[1], 3], np.int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = seg_output[i, j]
            image[i, j, :] = color[label, :]
    return image

