import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils

class Generator_net:
    """
        Neural Networks for Generator
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

        self.construct_computational_graph(training_batch_size, test_batch_size)

    def construct_computational_graph(training_batch_size = 32, test_batch_size = None):
        """
            construct tensorflow computational graph for neural networks (loss
            and train not included)
            we will have three group of computational graph here: training,
            testing(predict for a batch), and predict
            training_batch_size : int
            training_batch_size - size of a batch for training
            test_batch_size : int
            test_batch_size - size of a batch for testing
        """
        self.batch_size_train = training_batch_size
        self.batch_size_test = test_batch_size
        if (training_batch_size is None):
            self.deconv3_train = None
        else:
            self.img_stacked_input_train_placeholder = tf.placeholder(tf.float32, shape=[training_batch_size, self.img_height, self.img_width, self.num_stack * self.num_ch])
            self.frame_next_img_train_placeholder = tf.placeholder(tf.float32, shape=[training_batch_size, self.img_height, self.img_width, self.num_ch])
            self.action_input_train_placeholder = tf.placeholder(tf.float32, shape=[training_batch_size, self.num_act])

            self.conv1_train = tf.nn.conv2d(self.img_stacked_input_train_placeholder, self.w_conv1, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv1
            self.conv1_relu_train = tf.nn.relu(self.conv1_train)

            self.conv2_train = tf.nn.conv2d(self.conv1_relu_train, self.w_conv2, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv2
            self.conv2_relu_train = tf.nn.relu(self.conv2_train)

            self.conv3_train = tf.nn.conv2d(self.conv2_relu_train, self.w_conv3, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv3
            self.conv3_relu_train = tf.nn.relu(self.conv3_relu_train)

            conv3_relu_size = int(self.conv3_relu_train.shape[1]) * int(self.conv3_relu_train.shape[2]) * int(self.conv3_relu_train.shape[3])
            self.conv3_flat_train = tf.reshape(self.conv3_relu_train, [-1, conv3_relu_size])

            self.fc1_train = tf.matmul(self.conv3_flat_train, self.w_fc1) + self.b_fc1
            self.fc1_relu_train = tf.nn.relu(self.fc1_train)

            self.fc2_train = tf.matmul(self.fc1_relu_train, self.w_fc2)
            self.fca_train = tf.matmul(self.action_input_train_placeholder, self.w_fca)

            self.fc3_train = tf.multiply(self.fc2_train, self.fca_train) + self.b_fc3
            self.fc4_train = tf.matmul(self.fc3_train, self.w_fc4) + self.b_fc4

            self.fc5_train = tf.matmul(self.fc4_train, self.w_fc5) + self.b_fc5
            self.fc5_relu_train = tf.nn.relu(self.fc5_train)
            self.fc5_shaped_train = tf.reshape(self.fc5_relu_train, [-1, 10, 10, 64])

            self.deconv1_train = tf.nn.conv2d_transpose(self.fc5_shaped_train, self.w_deconv1, output_shape=[training_batch_size, 20, 20, 64], strides=[1, 2, 2, 1], padding='SAME') + self.b_deconv1
            self.deconv1_relu_train = tf.nn.relu(self.deconv1_train)

            self.deconv2_train = tf.nn.conv2d_transpose(self.deconv1_relu_train, self.w_deconv2, output_shape=[training_batch_size, 40, 40, 64], strides=[1, 2, 2, 1], padding='SAME') + self.b_deconv2
            self.deconv2_relu_train = tf.nn.relu(self.deconv2_train)

            self.deconv3_train = tf.nn.conv2d_transpose(self.deconv2_relu_train, self.w_deconv3, output_shape=[training_batch_size, 84, 84, 1], strides=[1, 2, 2, 1], padding='VALID') + self.b_deconv3

        if (test_batch_size is None):
            self.deconv3_test = None
        else:
            self.img_stacked_input_test_placeholder = tf.placeholder(tf.float32, shape=[test_batch_size, self.img_height, self.img_width, self.num_stack * self.num_ch])
            self.frame_next_img_test_placeholder = tf.placeholder(tf.float32, shape=[test_batch_size, self.img_height, self.img_width, self.num_ch])
            self.action_input_test_placeholder = tf.placeholder(tf.float32, shape=[test_batch_size, self.num_act])

            self.conv1_test = tf.nn.conv2d(self.img_stacked_input_test_placeholder, self.w_conv1, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv1
            self.conv1_relu_test = tf.nn.relu(self.conv1_test)

            self.conv2_test = tf.nn.conv2d(self.conv1_relu_test, self.w_conv2, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv2
            self.conv2_relu_test = tf.nn.relu(self.conv2_test)

            self.conv3_test = tf.nn.conv2d(self.conv2_relu_test, self.w_conv3, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv3
            self.conv3_relu_test = tf.nn.relu(self.conv3_relu_test)

            conv3_relu_size = int(self.conv3_relu_test.shape[1]) * \
                int(self.conv3_relu_test.shape[2]) * int(self.conv3_relu_test.shape[3])
            self.conv3_flat_test = tf.reshape(self.conv3_relu_test, [-1, conv3_relu_size])

            self.fc1_test = tf.matmul(self.conv3_flat_test, self.w_fc1) + self.b_fc1
            self.fc1_relu_test = tf.nn.relu(self.fc1_test)

            self.fc2_test = tf.matmul(self.fc1_relu_test, self.w_fc2)
            self.fca_test = tf.matmul(self.action_input_test_placeholder, self.w_fca)

            self.fc3_test = tf.multiply(self.fc2_test, self.fca_test) + self.b_fc3
            self.fc4_test = tf.matmul(self.fc3_test, self.w_fc4) + self.b_fc4

            self.fc5_test = tf.matmul(self.fc4_test, self.w_fc5) + self.b_fc5
            self.fc5_relu_test = tf.nn.relu(self.fc5_test)
            self.fc5_shaped_test = tf.reshape(self.fc5_relu_test, [-1, 10, 10, 64])

            self.deconv1_test = tf.nn.conv2d_transpose(self.fc5_shaped_test,\
                                                       self.w_deconv1,\
                                                       output_shape=[test_batch_size, 20, 20, 64], strides=[1, 2, 2, 1], padding='SAME') + self.b_deconv1
            self.deconv1_relu_test = tf.nn.relu(self.deconv1_test)

            self.deconv2_test = tf.nn.conv2d_transpose(self.deconv1_relu_test,\
                                                       self.w_deconv2,\
                                                       output_shape=[test_batch_size, 40, 40, 64], strides=[1, 2, 2, 1], padding='SAME') + self.b_deconv2
            self.deconv2_relu_test = tf.nn.relu(self.deconv2_test)

            self.deconv3_test = tf.nn.conv2d_transpose(self.deconv2_relu_test,
                                                       self.w_deconv3,
                                                       output_shape=[test_batch_size, 84, 84, 1], strides=[1, 2, 2, 1], padding='VALID') + self.b_deconv3

        self.img_stacked_input_predict_placeholder = tf.placeholder(tf.float32,\
                                                                    shape=[1, self.img_height, self.img_width, self.num_stack * self.num_ch])
        self.frame_next_img_predict_placeholder = tf.placeholder(tf.float32,\
                                                                 shape=[1, self.img_height, self.img_width, self.num_ch])
        self.action_input_predict_placeholder = tf.placeholder(tf.float32,\
                                                               shape=[1, self.num_act])

        self.conv1_predict = tf.nn.conv2d(self.img_stacked_input_predict_placeholder, self.w_conv1, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv1
        self.conv1_relu_predict = tf.nn.relu(self.conv1_predict)

        self.conv2_predict = tf.nn.conv2d(self.conv1_relu_predict, self.w_conv2, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv2
        self.conv2_relu_predict = tf.nn.relu(self.conv2_predict)

        self.conv3_predict = tf.nn.conv2d(self.conv2_relu_predict, self.w_conv3, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv3
        self.conv3_relu_predict = tf.nn.relu(self.conv3_relu_predict)

        conv3_relu_size = int(self.conv3_relu_predict.shape[1]) * \
                int(self.conv3_relu_predict.shape[2]) * \
                int(self.conv3_relu_predict.shape[3])
        self.conv3_flat_predict = tf.reshape(self.conv3_relu_predict, [-1, conv3_relu_size])

        self.fc1_predict = tf.matmul(self.conv3_flat_predict, self.w_fc1) + self.b_fc1
        self.fc1_relu_predict = tf.nn.relu(self.fc1_predict)

        self.fc2_predict = tf.matmul(self.fc1_relu_predict, self.w_fc2)
        self.fca_predict = tf.matmul(self.action_input_predict_placeholder, self.w_fca)

        self.fc3_predict = tf.multiply(self.fc2_predict, self.fca_predict) + self.b_fc3
        self.fc4_predict = tf.matmul(self.fc3_predict, self.w_fc4) + self.b_fc4

        self.fc5_predict = tf.matmul(self.fc4_predict, self.w_fc5) + self.b_fc5
        self.fc5_relu_predict = tf.nn.relu(self.fc5_predict)
        self.fc5_shaped_predict = tf.reshape(self.fc5_relu_predict, [-1, 10, 10, 64])

        self.deconv1_predict = tf.nn.conv2d_transpose(self.fc5_shaped_predict,\
                                                    self.w_deconv1,\
                                                    output_shape=[1, 20, 20, 64], strides=[1, 2, 2, 1], padding='SAME') + self.b_deconv1
        self.deconv1_relu_predict = tf.nn.relu(self.deconv1_predict)

        self.deconv2_predict = tf.nn.conv2d_transpose(self.deconv1_relu_predict,\
                                                    self.w_deconv2,\
                                                    output_shape=[1, 40, 40, 64], strides=[1, 2, 2, 1], padding='SAME') + self.b_deconv2
        self.deconv2_relu_predict = tf.nn.relu(self.deconv2_predict)

        self.deconv3_predict = tf.nn.conv2d_transpose(self.deconv2_relu_predict,
                                                    self.w_deconv3,\
                                                    output_shape=[1, 84, 84, 1], strides=[1, 2, 2, 1], padding='VALID') + self.b_deconv3



