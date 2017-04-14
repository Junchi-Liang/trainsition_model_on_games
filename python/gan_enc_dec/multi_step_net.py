import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils

class Multi_Step_net:
    """
        Neural Networks with multi-step prediction
    """
    def __init__(self, stacked_img_num = 4, image_height = 84, image_width =\
                 84, num_actions = 18, num_channels = 1, training_batch_size =\
                32, test_batch_size = None):
        """
            stacked_img_num : int
            stacked_img_num - number of stakced images
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

        self.w_fc5 = nn_utils.cnn_utils.normal_weight_variable([1024, 6400], 0.1)
        self.b_fc5 = nn_utils.cnn_utils.bias_variable([6400], 0.1)

        self.w_deconv1 = nn_utils.cnn_utils.weight_deconvolution_normal([6, 6], 64, 64, 0.1)
        self.b_deconv1 = nn_utils.cnn_utils.bias_convolution(64, 0.1)

        self.w_deconv2 = nn_utils.cnn_utils.weight_deconvolution_normal([6, 6], 64, 64, 0.1)
        self.b_deconv2 = nn_utils.cnn_utils.bias_convolution(64, 0.1)

        self.w_deconv3 = nn_utils.cnn_utils.weight_deconvolution_normal([6, 6], 64, 1, 0.1)
        self.b_deconv3 = nn_utils.cnn_utils.bias_convolution(1, 0.1)

        self.param = [self.w_conv1, self.b_conv1, self.w_conv2, self.b_conv2, self.w_conv3, self.b_conv3,\
                      self.w_fc1, self.b_fc1, self.w_fc2, self.w_fca, self.b_fc3, self.w_fc4, self.b_fc4,\
                      self.w_fc5, self.b_fc5, self.w_deconv1, self.b_deconv1, self.w_deconv2, self.b_deconv2,\
                      self.w_deconv3, self.b_deconv3]

        self.net_train = self.construct_network_computational_graph(training_batch_size)
        self.net_predict = self.construct_network_computational_graph(1)
        if  (not(test_batch_size is None)):
            self.net_test = self.construct_network_computational_graph(test_batch_size)
        self.train_loss = self.mean_square_loss(self.net_train["deconv3"], self.net_train["frame_next_img_placeholder"])
        self.train_step = tf.train.RMSPropOptimizer(1e-4,
                                                    momentum=0.9).minimize(self.train_loss, var_list=self.param)


    def construct_network_computational_graph(self, input_layer = None, params_shared = None, batch_size):
        """
            construct tensorflow computational graph for neural networks (loss
            and train step not included)
            batch_size : int
            batch_size - size of batch
            return layers
            layers : dictionary of tensorflow.python.framework.ops.Tensor
            layers - all layers of a network
        """
        params = {}
        if (params_shared is None):
            params["w_conv1"] = nn_utils.cnn_utils.weight_convolution_normal([6, 6], stacked_img_num, 64, 0.1)
            params["b_conv1"] = nn_utils.cnn_utils.bias_convolution(64, 0.1)

            params["w_conv2"] = nn_utils.cnn_utils.weight_convolution_normal([6, 6], 64, 64, 0.1)
            params["b_conv2"] = nn_utils.cnn_utils.bias_convolution(64, 0.1)

            params["w_conv3"] = nn_utils.cnn_utils.weight_convolution_normal([6, 6], 64, 64, 0.1)
            params["b_conv3"] = nn_utils.cnn_utils.bias_convolution(64, 0.1)

            params["w_fc1"] = nn_utils.cnn_utils.normal_weight_variable([6400, 1024], 0.1)
            params["b_fc1"] = nn_utils.cnn_utils.bias_variable([1024], 0.1)
        
            params["w_fc2"] = nn_utils.cnn_utils.normal_weight_variable([1024, 2048], 0.1)
            params["w_fca"] = nn_utils.cnn_utils.normal_weight_variable([18, 2048], 0.1)

            params["b_fc3"] = nn_utils.cnn_utils.bias_variable([2048], 0.1)

            params["w_fc4"] = nn_utils.cnn_utils.normal_weight_variable([2048, 1024], 0.1)
            params["b_fc4"] = nn_utils.cnn_utils.bias_variable([1024], 0.1)

            params["w_fc5"] = nn_utils.cnn_utils.normal_weight_variable([1024, 6400], 0.1)
            params["b_fc5"] = nn_utils.cnn_utils.bias_variable([6400], 0.1)

            params["w_deconv1"] = nn_utils.cnn_utils.weight_deconvolution_normal([6, 6], 64, 64, 0.1)
            params["b_deconv1"] = nn_utils.cnn_utils.bias_convolution(64, 0.1)

            params["w_deconv2"] = nn_utils.cnn_utils.weight_deconvolution_normal([6, 6], 64, 64, 0.1)
            params["b_deconv2"] = nn_utils.cnn_utils.bias_convolution(64, 0.1)

            params["w_deconv3"] = nn_utils.cnn_utils.weight_deconvolution_normal([6, 6], 64, 1, 0.1)
            params["b_deconv3"] = nn_utils.cnn_utils.bias_convolution(1, 0.1)
        else:
            params = params_shared
        layers = {}
        if (input_layers is None):
            layers["img_stacked_input_placeholder"] = tf.placeholder(tf.float32,\
                                                        shape=[batch_size, self.img_height,\
                                                        self.img_width, self.num_stack *\
                                                        self.num_ch])
        else:
            layers["img_stacked_input_placeholder"] = input_layer
        layers["frame_next_img_placeholder"] = tf.placeholder(tf.float32,
                                                                    shape=[batch_size,
                                                                           self.img_height,
                                                                           self.img_width,
                                                                           self.num_ch])
        layers["action_input_placeholder"] = tf.placeholder(tf.float32,
                                                            shape=[batch_size,
                                                                   self.num_act])
        layers["conv1"] = tf.nn.conv2d(layers["img_stacked_input_placeholder"],
                                       params["w_conv1"], strides=[1, 2, 2, 1], padding='VALID') + params["b_conv1"]
        layers["conv1_relu"] = tf.nn.relu(layers["conv1"])
        layers["conv2"] = tf.nn.conv2d(layers["conv1_relu"], params["w_conv2"],
                                       strides=[1, 2, 2, 1], padding='SAME') + params["b_conv2"]
        layers["conv2_relu"] = tf.nn.relu(layers["conv2"])
        layers["conv3"] = tf.nn.conv2d(layers["conv2_relu"], params["w_conv3"],
                                       strides=[1, 2, 2, 1], padding='SAME') + params["b_conv3"]
        layers["conv3_relu"] = tf.nn.relu(layers["conv3"])
        conv3_relu_size = int(layers["conv3_relu"].shape[1]) *\
                int(layers["conv3_relu"].shape[2]) *\
                int(layers["conv3_relu"].shape[3])
        layers["conv3_flat"] = tf.reshape(layers["conv3_relu"], [-1,
                                                                  conv3_relu_size])
        layers["fc1"] = tf.matmul(layers["conv3_flat"], params["w_fc1"]) + params["b_fc1"]
        layers["fc1_relu"] = tf.nn.relu(layers["fc1"])
        layers["fc2"] = tf.matmul(layers["fc1_relu"], params["w_fc2"])
        layers["fca"] = tf.matmul(layers["action_input_placeholder"], params["w_fca"])
        layers["fc3"] = tf.multiply(layers["fc2"], layers["fca"]) + params["b_fc3"]
        layers["fc4"] = tf.matmul(layers["fc3"], params["w_fc4"]) + params["b_fc4"]
        layers["fc5"] = tf.matmul(layers["fc4"], params["w_fc5"]) + params["b_fc5"]
        layers["fc5_relu"] = tf.nn.relu(layers["fc5"])
        layers["fc5_shaped"] = tf.reshape(layers["fc5_relu"], [-1, 10, 10, 64])
        layers["deconv1"] = tf.nn.conv2d_transpose(layers["fc5_shaped"],
                                                   params["w_deconv1"],
                                                   output_shape=[batch_size,
                                                                 20, 20, 64],
                                                   strides=[1, 2, 2, 1],
                                                   padding='SAME') + params["b_deconv1"]
        layers["deconv1_relu"] = tf.nn.relu(layers["deconv1"])
        layers["deconv2"] = tf.nn.conv2d_transpose(layers["deconv1_relu"],
                                                   params["w_deconv2"],
                                                   output_shape=[batch_size,
                                                                 40, 40, 64],
                                                   strides=[1, 2, 2, 1],
                                                   padding='SAME') + params["b_deconv2"]
        layers["deconv2_relu"] = tf.nn.relu(layers["deconv2"])
        layers["deconv3"] = tf.nn.conv2d_transpose(layers["deconv2_relu"],
                                                   params["w_deconv3"],
                                                   output_shape=[batch_size,
                                                                 84, 84, 1],
                                                   strides=[1, 2, 2, 1],
                                                   padding='VALID') + params["b_deconv3"]
        return [layers, params]

    def mean_square_loss(self, net_output, true_ans):
        """
            construct a computational graph for mean square loss
            net_output : tensorflow.python.framework.ops.Tensor
            net_output - a computational graph node for network output
            true_ans : tensorflow.python.framework.ops.Tensor
            true_ans - a computational graph node for true answer, it should be
            a placeholder
            return loss
            loss : tensorflow.python.framework.ops.Tensor
            loss - a computational node for loss
        """
        return tf.reduce_mean(tf.square(net_output - true_ans))

    def gradient_difference_loss(self, net_output, true_ans):
        """
            construct a computational graph for gradient difference loss
            net_output : tensorflow.python.framework.ops.Tensor
            net_output - a computational graph node for network output
            true_ans : tensorflow.python.framework.ops.Tensor
            true_ans - a computational graph node for true answer, it should be
            a placeholder
            return loss
            loss : tensorflow.python.framework.ops.Tensor
            loss - a computational node for loss
        """
        g1 = np.float32(np.array([[0, -1, 0], [0, 1 ,0], [0, 0, 0]])).reshape([3, 3, 1])
        g2 = np.float32(np.array([[0, 0, 0], [0, 1 ,0], [0, -1, 0]])).reshape([3, 3, 1])
        g3 = np.float32(np.array([[0, 0, 0], [-1, 1 ,0], [0, 0, 0]])).reshape([3, 3, 1])
        g4 = np.float32(np.array([[0, 0, 0], [0, 1 ,-1], [0, 0, 0]])).reshape([3, 3, 1])
        g = np.stack((g1, g2, g3, g4), axis = 3)
        net_output_gradient = tf.nn.conv2d(net_output, g, strides= [1, 1, 1, 1], padding='VALID')
        ground_truth = tf.nn.conv2d(true_ans, g, strides = [1, 1, 1, 1], padding='VALID')
        loss = tf.reduce_mean(tf.square(net_output_gradient - ground_truth))
        return loss

