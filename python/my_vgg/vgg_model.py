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

    def construct_network_computation_graph(input_layer = None, batch_size = 0, shared_weight = None):
        """
            construct computation graph of Tensorflow for the networks
            input_layer : tensorflow.python.framework.ops.Tensor
            input_layer = input layer. If it is None, a new layer of placeholder will be constructed
            batch_size : int
            batch_size = batch size. If an input layer is provided, this argument will be omitted
            shared_weight : dictionary
            shared_weight = weights for each layers. If it is None, new weights will be constructed
            ---------------------------------------------------------------------------------------
            return [layers, parameters]
        """
        if (shared_weight is None):
            parameters = {}
            for i in range(1, 14):
                parameters["w_conv" + str(i)] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], self.image_channel, 64, 0.1)
                parameters["b_covn" + str(i)] = nn_utils.cnn_utils.bias_convolution(64, 0.0)
        else:
            parameters = shared_weight
