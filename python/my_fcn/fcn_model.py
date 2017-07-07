import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils
from my_vgg.vgg16_model import VGG16_model

class FCN_model:
    """
        My Implementation for FCN-8s
    """
    def __init__(self, convert_from_vgg = None, load_from_vgg = None):
        """
            convert_from_vgg : VGG16_model
            convert_from_vgg = the VGG16 model which will be converted to FCN.
                               when an VGG16 model is converted to FCN, they will share the same weight object,
                               i.e. any modification in these weights will appear in both models
            load_from_vgg : VGG16_model
            load_from_vgg = the VGG16 model model which will be loaded to FCN.
                            when an VGG16 model is loaded to FCN, they will keep their own saparate weight objects,
                            i.e. changes of one model will only appear in this model
        """

    def build_computation_graph(self, img_height, img_width, img_channel, parameters, batch_size, num_class, drop_out_prob, input_layer = None):
        """
            build computation graph for the FCN architecture
            assume color images with RGB channels
            img_height : int
            img_height = height of images
            img_width : int
            img_width = width of images
            img_channel : int
            img_channel = number of channels in images
            parameters : dictionary
            parameters = parameters used in this architecture
            batch_size : int
            batch_size = batch size for this graph
            num_class : int
            num_class = number of classes for output layer
            drop_out_prob : float
            drop_out_prob = probability for drop out
            input_layer : tensorflow.python.framework.ops.Tensor
            input_layer = input layer. If it is None, a new layer of placeholder will be constructed
            ------------------------------------------------
            return layers
            layers : dictionary
            layers = collection of tensors for each layers, indexed by name
        """
        layers = []
        if (input_layer is None):
            layers["image_input"] = tf.placeholder(tf.float32, shape = [\
                                       batch_size, img_height, img_width, img_channel])
        else:
            layers["image_input"] = input_layer
        layers["conv1_1"] = tf.nn.bias_add(tf.nn.conv2d(layers["image_input"],
                                       parameters["w_conv1_1"], strides=[1, 1, 1, 1], padding='SAME'), parameters["b_conv1_1"])
        layers["relu1_1"] = tf.nn.relu(layers["conv1_1"])
        layers["conv1_2"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu1_1"],
                                       parameters["w_conv1_2"], strides=[1, 1, 1, 1], padding='SAME'), parameters["b_conv1_2"])
        layers["relu1_2"] = tf.nn.relu(layers["conv1_2"])
        layers["pool1"] = tf.nn.max_pool(layers["relu1_2"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv2_1"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool1"], parameters["w_conv2_1"], strides = [1, 1 ,1, 1],\
                                       padding = 'SAME'), parameters["b_conv2_1"])
        layers["relu2_1"] = tf.nn.relu(layers["conv2_1"])
        layers["conv2_2"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu2_1"], parameters["w_conv2_2"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv2_2"])
        layers["relu2_2"] = tf.nn.relu(layers["conv2_2"])
        layers["pool2"] = tf.nn.max_pool(layers["relu2_2"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv3_1"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool2"], parameters["w_conv3_1"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv3_1"])
        layers["relu3_1"] = tf.nn.relu(layers["conv3_1"])
        layers["conv3_2"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu3_1"], parameters["w_conv3_2"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv3_2"])
        layers["relu3_2"] = tf.nn.relu(layers["conv3_2"])
        layers["conv3_3"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu3_2"], parameters["w_conv3_3"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv3_3"])
        layers["relu3_3"] = tf.nn.relu(layers["conv3_3"])
        layers["pool3"] = tf.nn.max_pool(layers["relu3_3"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv4_1"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool3"], parameters["w_conv4_1"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv4_1"])
        layers["relu4_1"] = tf.nn.relu(layers["conv4_1"])
        layers["conv4_2"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu4_1"], parameters["w_conv4_2"], strides = [1, 1, 1, 1], \
                                       padding = 'SAME'), parameters["b_conv4_2"])
        layers["relu4_2"] = tf.nn.relu(layers["conv4_2"])
        layers["conv4_3"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu4_2"], parameters["w_conv4_3"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv4_3"])
        layers["relu4_3"] = tf.nn.relu(layers["conv4_3"])
        layers["pool4"] = tf.nn.max_pool(layers["relu4_3"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv5_1"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool4"], parameters["w_conv5_1"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv5_1"])
        layers["relu5_1"] = tf.nn.relu(layers["conv5_1"])
        layers["conv5_2"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu5_1"], parameters["w_conv5_2"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv5_2"])
        layers["relu5_2"] = tf.nn.relu(layers["conv5_2"])
        layers["conv5_3"] = tf.nn.bias_add(tf.nn.conv2d(layers["relu5_2"], parameters["w_conv5_3"], strides = [1, 1, 1, 1], \
                                        padding = 'SAME'), parameters["b_conv5_3"])
        layers["relu5_3"] = tf.nn.relu(layers["conv5_3"])
        layers["pool5"] = tf.nn.max_pool(layers["relu5_3"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        layers["conv6"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool5"], parameters["w_conv6"], strides = [1, 1, 1, 1], \
                                         padding = 'SAME'), parameters["b_conv6"])
        layers["relu6"] = tf.nn.relu(layers["conv6"])
        layers["dropout6"] = tf.nn.dropout(layers["relu6"], drop_out_prob)
        layers["conv7"] = tf.nn.bias_add(tf.nn.conv2d(layers["dropout6"], parameters["w_conv7"], strides = [1, 1, 1, 1], \
                                         padding = 'SAME'), parameters["b_conv7"])
        layers["relu7"] = tf.nn.relu(layers["conv7"])
        layers["dropout7"] = tf.nn.dropout(layers["relu7"], drop_out_prob)
        layers["score_up1"] = tf.nn.bias_add(tf.nn.conv2d(layers["dropout7"], parameters["w_score_up1"], strides = [1, 1, 1, 1], \
                                          padding = 'SAME'), parameters["b_score_up1"])
        pool4_shape = [int(layers["pool4"].get_shape()[0]), int(layers["pool4"].get_shape()[1]), \
                       int(layers["pool4"].get_shape()[2]), int(layers["pool4"].get_shape()[3])]
        layers["score_up2"] = tf.nn.bias_add(tf.nn.conv2d_transpose(layers["score_up1"], parameters["w_score_up2"], \
                                            output_shape = [pool4_shape[0], pool4_shape[1], pool4_shape[2], num_class], \
                                            strides = [1, 2, 2, 1], padding = 'SAME'), parameters["b_score_up2"])
        layers["score_pool4"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool4"], parameters["w_score_pool4"], \
                                               strides = [1, 1, 1, 1], padding = 'SAME'), parameters["b_score_pool4"])
        layers["fuse_pool4"] = tf.add(layers["scoreup2"], layers["score_pool4"])
        pool3_shape = [int(layers["pool3"].get_shape()[0]), int(layers["pool3"].get_shape()[1]), \
                       int(layers["pool3"].get_shape()[2]), int(layers["pool3"].get_shape()[3])]
        layers["score_up4"] = tf.nn.bias_add(tf.nn.conv2d_transpose(layers["fuse_pool4"], parameters["w_score_up4"], \
                                            output_shape = [pool3_shape[0], pool3_shape[1], pool3_shape[2], num_class], \
                                            strides = [1, 2, 2, 1], padding = 'SAME'), parameters["b_score_up4"])
        layers["score_pool3"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool3"], parameters["w_score_pool3"], \
                                               strides = [1, 1, 1, 1], padding = 'SAME'), parameters["b_score_pool3"])
        layers["fuse_pool3"] = tf.add(layers["score_pool3"], layers["scoreup4"])
        layers["score_output"] = tf.nn.bias_add(tf.nn.conv2d_transpose(layers["fuse_pool3"], parameters["w_score_output"], \
                                                output_shape = [batch_size, img_height, img_width, num_class], \
                                                strides = [1, 8, 8, 1], padding = 'SAME'), parameters["b_score_output"])
        return layers

    def convert_VGG(self, vgg_model, num_class):
        """
            convert weights from VGG model to FCN
            vgg_model : VGG16_model
            vgg_model = the vgg model which will be converted
            num_class : int
            num_class = number of output classes
            ---------------------------------------------------
            return parameters
            parameters : dictionary
            parameters = collection of parameters used in this architecture, indexed by name
        """
        
