import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils

class VGG_model:
    """
        My Implementation for VGG
    """
    def __init__(self, img_height, img_width, img_channel, training_batch_size = None, test_batch_size = None):
        """
            img_height : int
            img_height = height of images
            img_width : int
            img_width = width of images
            img_channel : int
            img_channel = number of channels in images
            training_batch_size : int
            training_batch_size = size of training batch, when it is none, no training network is constructed
            test_batch_size : int
            test_batch_size = size of test batch, when it is none, no test network is constructed
        """
        self.image_height = img_height
        self.image_width = img_width
        self.image_channel = img_channel
        self.train_batch_size = training_batch_size
        self.test_batch_size = test_batch_size

