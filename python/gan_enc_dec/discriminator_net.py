import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils

class Discriminator_net:
    """
        Neural Networks for Discriminator 
    """
    def __init__(self, stacked_img_num = 5, image_height = 84, image_width = 84, num_actions = 18, num_channels = 1, training_batch_size = 32, test_batch_size = None):
        """
            stacked_img_num : int
            stacked_img_num - number of stacked image for input
            image_height : int
            image_height - height of the image
            image_width : int
            image_width - width of the image
            num_actions : int
            num_actions - number of actions
            num_channels : int
            num_channels - number of channels
        """
        self.img_height = image_height
        self.img_width = image_width
        self.num_act = num_actions
        self.num_stack = stacked_img_num
        self.num_ch = num_channels
        self.w_conv1 = nn_utils.cnn_utils.weight_convolution_normal([6, 6], stacked_img_num, 64, 0.1)
        self.b_conv1 = nn_utils.cnn_utils.bias_convolution(64, 0.1)

        self.w_conv2 = nn_utils.cnn_utils.weight_convolution_normal([6, 6], 64, 64, 0.1)
        self.b_conv2 = nn_utils.cnn_utils.bias_convolution(64, 0.1)

        self.w_conv3 = nn_utils.cnn_utils.weight_convolution_normal([6, 6], 64, 64, 0.1)
        self.b_conv3 = nn_utils.cnn_utils.bias_convolution(64, 0.1)

        self.w_fc1 = nn_utils.cnn_utils.normal_weight_variable([6400, 1024], 0.1)
        self.b_fc1 = nn_utils.cnn_utils.bias_variable([1024], 0.1)
        
        self.w_fc2 = nn_utils.cnn_utils.normal_weight_variable([1024, 2048], 0.1)
        self.w_fca = nn_utils.cnn_utils.normal_weight_variable([18, 2048], 0.1)

        self.b_fc3 = nn_utils.cnn_utils.bias_variable([2048], 0.1)

        self.w_fc4 = nn_utils.cnn_utils.normal_weight_variable([2048, 1024], 0.1)
        self.b_fc4 = nn_utils.cnn_utils.bias_variable([1024], 0.1)

        self.w_fc5 = nn_utils.cnn_utils.normal_weight_variable([1024, 256], 0.1)
        self.b_fc5 = nn_utils.cnn_utils.bias_variable([256], 0.1)

        self.w_fc6 = nn_utils.cnn_utils.normal_weight_variable([256, 1], 0.1)
        self.b_fc6 = nn_utils.cnn_utils.bias_variable([1], 0.1)

        self.param = [self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2, self.w_conv3, self.b_conv3,\
                      self.w_fc1, self.b_fc1, self.w_fc2, self.w_fca, self.b_fc3, self.w_fc4, self.w_fc5,\
                      self.b_fc5, self.w_fc6, self.b_fc6]

        self.net_train = self.construct_network_computational_graph(training_batch_size)
        self.net_predict = self.construct_network_computational_graph(1)
        if (not(test_batch_size is None)):
            self.net_test = self.construct_network_computational_graph(test_batch_size)
        self.train_loss = tf.nn.weighted_cross_entropy_with_logits(self.net_train["true_label_placeholder"], self.net_train["fc6"], 1.0)
        self.train_step = tf.train.RMSPropOptimizer(1e-4,
                                                    momentum=0.9).minimize(self.train_loss)

    def construct_network_computational_graph(self, batch_size):
        """
            construct tensorflow computational graph for neural networks (loss
            and train step not included)
            batch_size : int
            batch_size - size of batch
            return layers
            layers : dictionary of tensorflow.python.framework.ops.Tensor
            layers - all layers of a network
        """
        layers = {}
        layers["img_stacked_input_placeholder"] = tf.placeholder(tf.float32,\
                                                        shape=[batch_size, self.img_height,\
                                                        self.img_width, self.num_stack *\
                                                        self.num_ch])
        layers["true_label_placeholder"] = tf.placeholder(tf.float32,
                                                                    shape=[batch_size, 1])
        layers["action_input_placeholder"] = tf.placeholder(tf.float32,
                                                            shape=[batch_size,
                                                                   self.num_act])
        layers["conv1"] = tf.nn.conv2d(layers["img_stacked_input_placeholder"],
                                       self.w_conv1, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv1
        layers["conv1_relu"] = tf.nn.relu(layers["conv1"])
        layers["conv2"] = tf.nn.conv2d(layers["conv1_relu"], self.w_conv2,
                                       strides=[1, 2, 2, 1], padding='SAME') + self.b_conv2
        layers["conv2_relu"] = tf.nn.relu(layers["conv2"])
        layers["conv3"] = tf.nn.conv2d(layers["conv2_relu"], self.w_conv3,
                                       strides=[1, 2, 2, 1], padding='SAME') + self.b_conv3
        layers["conv3_relu"] = tf.nn.relu(layers["conv3"])
        conv3_relu_size = int(layers["conv3_relu"].shape[1]) *\
                int(layers["conv3_relu"].shape[2]) *\
                int(layers["conv3_relu"].shape[3])
        layers["conv3_flat"] = tf.reshape(layers["conv3_relu"], [-1,
                                                                  conv3_relu_size])
        layers["fc1"] = tf.matmul(layers["conv3_flat"], self.w_fc1) + self.b_fc1
        layers["fc1_relu"] = tf.nn.relu(layers["fc1"])
        layers["fc2"] = tf.matmul(layers["fc1_relu"], self.w_fc2)
        layers["fca"] = tf.matmul(layers["action_input_placeholder"], self.w_fca)
        layers["fc3"] = tf.multiply(layers["fc2"], layers["fca"]) + self.b_fc3
        layers["fc4"] = tf.matmul(layers["fc3"], self.w_fc4) + self.b_fc4
        layers["fc5"] = tf.matmul(layers["fc4"], self.w_fc5) + self.b_fc5
        layers["fc5_relu"] = tf.nn.relu(layers["fc5"])
        layers["fc6"] = tf.matmul(layers["fc5_relu"], self.w_fc6) + self.b_fc6
        layers["output"] = tf.sigmoid(layers["fc6"])

        return layers


