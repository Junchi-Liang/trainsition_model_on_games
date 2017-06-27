import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils

class VGG16_model:
    """
        My Implementation for VGG16
    """
    def __init__(self, img_height, img_width, img_channel, training_batch_size = None, test_batch_size = None, pretrained_weight = None):
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
            pretrained_weight : dictionary
            pretrained_weight = weights from a pretrained model
        """
        self.image_height = img_height
        self.image_width = img_width
        self.image_channel = img_channel
        self.train_batch_size = training_batch_size
        self.test_batch_size = test_batch_size
        if (training_batch_size is not None):
            self.train_net, self.parameters = construct_network_computation_graph(batch_size = training_batch_size, shared_weight = pretrained_weight)
        if (test_batch_size is not None):
            try:
                self.test_net, self.parameters = construct_network_computation_graph(batch_size = test_batch_size, shared_weight = self.parameters)
            except AttributeError:
                self.test_net, self.parameters = construct_network_computation_graph(batch_size = test_batch_size, shared_weight = pretrained_weight)

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
                parameters["b_conv" + str(i)] = nn_utils.cnn_utils.bias_convolution(64, 0.0)
            parameters["w_fc2"] = nn_utils.cnn_utils.normal_weight_variable([4096, 4096], 0.1)
            parameters["b_fc2"] = nn_utils.cnn_utils.bias_variable([4096], 1.0)
            parameters["w_fc3"] = nn_utils.cnn_utils.normal_weight_variable([4096, 1000], 0.1)
            parameters["b_fc3"] = nn_utils.cnn_utils.bias_variable([1000], 1.0)
        else:
            parameters = shared_weight
        layers = {}
        if (input_layer is None):
            layers["image_input"] = tf.placeholder(tf.float32, shape = [\
                                       batch_size, self.image_height, self.image_width, self.image_channel])
        else:
            layers["image_input"] = input_layer
        layers["conv1"] = tf.nn.add_bias(tf.nn.conv2d(layers["image_input"],
                                       parameters["w_conv1"], strides=[1, 1, 1, 1], padding='SAME'), parameters["b_conv1"])
        layers["relu1"] = tf.nn.relu(layers["conv1"])
        layers["conv2"] = tf.nn.add_bias(tf.nn.conv2d(layers["conv1"],
                                       parameters["w_conv2"], strides=[1, 1, 1, 1], padding='SAME'), parameters["b_conv2"])
        layers["relu2"] = tf.nn.relu(layers["conv2"])
        layers["pool1"] = tf.nn.max_pool(layers["relu2"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv3"] = tf.nn.add_bias(tf.nn.conv2d(layers["pool1"], parameters["w_conv3"], strides = [1, 1 ,1, 1],\
                                       padding = 'SAME'), parameters["b_conv3"])
        layers["relu3"] = tf.nn.relu(layers["conv3"])
        layers["conv4"] = tf.nn.add_bias(tf.nn.conv2d(layers["relu3"], parameters["w_conv4"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv4"])
        layers["relu4"] = tf.n.relu(layers["conv4"])
        layers["pool2"] = tf.nn.max_pool(layers["relu4"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv5"] = tf.nn.add_bias(tf.nn.conv2d(layers["pool2"], parameters["w_conv5"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv5"])
        layers["relu5"] = tf.nn.relu(layers["conv5"])
        layers["conv6"] = tf.nn.add_bias(tf.nn.conv2d(layers["relu5"], parameters["w_conv6"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv6"])
        layers["relu6"] = tf.nn.relu(layers["conv6"])
        layers["conv7"] = tf.nn.add_bias(tf.nn.conv2d(layers["relu6"], parameters["w_conv7"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv7"])
        layers["relu7"] = tf.nn.relu(layers["conv7"])
        layers["pool3"] = tf.nn.max_pool(layers["relu7"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv8"] = tf.nn.add_bias(tf.nn.conv2d(layers["pool3"], parameters["w_conv8"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv8"])
        layers["relu8"] = tf.nn.relu(layers["conv8"])
        layers["conv9"] = tf.nn.add_bias(tf.nn.conv2d(layers["relu8"], parameters["w_conv9"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv9"])
        layers["relu9"] = tf.nn.relu(layers["conv9"])
        layers["conv10"] = tf.nn.add_bias(tf.nn.conv2d(layers["relu9"], parameters["w_conv10"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv10"])
        layers["relu10"] = tf.nn.relu(layers["conv10"])
        layers["pool4"] = tf.nn.max_pool(layers["relu10"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv11"] = tf.nn.add_bias(tf.nn.conv2d(layers["pool4"], parameters["w_conv11"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv11"])
        layers["relu11"] = tf.nn.relu(layers["conv11"])
        layers["conv12"] = tf.nn.add_bias(tf.nn.conv2d(layers["relu11"], parameters["w_conv12"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv12"])
        layers["relu12"] = tf.nn.relu(layers["conv12"])
        layers["conv13"] = tf.nn.add_bias(tf.nn.conv2d(layers["relu12"], parameters["w_conv13"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv13"])
        layers["relu13"] = tf.nn.relu(layers["conv13"])
        layers["pool5"] = tf.nn.max_pool(layers["relu13"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        pool5_size = int(layers["pool5"].shape[1]) * int(layers["pool5"].shape[2]) * int(layers["pool5"].shape[3])
        layers["flat"] = tf.reshape(layers["pool5"], [-1, pool5_size])
        if (shared_weight is None):
            parameters["w_fc1"] = nn_utils.cnn_utils.normal_weight_variable([pool5_size, 4096], 0.1)
            parameters["b_fc1"] = nn_utils.cnn_utils.bias_variable([4096], 1.0)
        layers["fc1"] = tf.nn.add_bias(tf.matmul(layers["pool5"], parameters["w_fc1"]), parameters["b_fc1"])
        layers["relu14"] = tf.nn.relu(layers["fc1"])
        layers["fc2"] = tf.nn.add_bias(tf.matmul(layers["relu14"], parameters["w_fc2"]), parameters["b_fc2"])
        layers["relu15"] = tf.nn.relu(layers["fc2"])
        layers["fc3"] = tf.nn.add_bias(tf.matmul(layers["relu15"], parameters["w_fc3"]), parameters["b_fc3"])
        layers["output"] = tf.nn.softmax(layers["fc3"])
        return [layers, parameters]
