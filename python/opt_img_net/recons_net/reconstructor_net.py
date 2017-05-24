import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils
import data_utils.data_read_util
import data_utils.img_process_util

class Reconstructor_net:
    """
        Neural networks constructing next frame from previous frames and optical flows
    """
    def __init__(self, stacked_img_num = 2, image_height = 84, image_width = 84, num_channels = 1, training_batch_size = 32):
        """
            stacked_img_num : int
            stacked_img_num - number of stakced images
            image_height : int
            image_height - height of the image
            image_width : int
            image_width - width of the image
            num_channels : int
            num_channels - number of channels of images
            training_batch_size : int
            training_batch_size - the size of training batch
        """

        self.img_stacked = stacked_img_num
        self.img_channel = num_channels
        self.img_height = image_height
        self.img_width = image_width

    def construct_network_computational_graph(self, batch_size, input_layer = None, params_shared = None):
        """
            construct tensorflow computational graph for neural networks (loss
            and train step not included)
            batch_size : int
            batch_size - size of batch
            return [layers, params]
            layers : dictionary of tensorflow.python.framework.ops.Tensor
            layers - all layers of a network
        """
        if (params_shared is None):
            params = {}
        else:
            params = params_shared
