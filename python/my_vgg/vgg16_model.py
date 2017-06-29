import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils

class VGG16_model:
    """
        My Implementation for VGG16
    """
    def __init__(self, img_height = 224, img_width = 224, img_channel = 3, training_batch_size = None, test_batch_size = None, pretrained_weight = None):
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
            self.train_net, self.parameters = self.construct_network_computation_graph(batch_size = training_batch_size, shared_weight = pretrained_weight)
        if (test_batch_size is not None):
            try:
                self.test_net, self.parameters = self.construct_network_computation_graph(batch_size = test_batch_size, shared_weight = self.parameters)
            except AttributeError:
                self.test_net, self.parameters = self.construct_network_computation_graph(batch_size = test_batch_size, shared_weight = pretrained_weight)

    def construct_network_computation_graph(self, input_layer = None, batch_size = 0, shared_weight = None):
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
            parameters["w_conv1_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], self.image_channel, 64, 0.1)
            parameters["b_conv1_1"] = nn_utils.cnn_utils.bias_convolution(64, 0.0)
            parameters["w_conv1_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 64, 64, 0.1)
            parameters["b_conv1_2"] = nn_utils.cnn_utils.bias_convolution(64, 0.0)
            parameters["w_conv2_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 64, 128, 0.1)
            parameters["b_conv2_1"] = nn_utils.cnn_utils.bias_convolution(128, 0.0)
            parameters["w_conv2_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 128, 128, 0.1)
            parameters["b_conv2_2"] = nn_utils.cnn_utils.bias_convolution(128, 0.0)
            parameters["w_conv3_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 128, 256, 0.1)
            parameters["b_conv3_1"] = nn_utils.cnn_utils.bias_convolution(256, 0.0)
            parameters["w_conv3_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 256, 256, 0.1)
            parameters["b_conv3_2"] = nn_utils.cnn_utils.bias_convolution(256, 0.0)
            parameters["w_conv3_3"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 256, 256, 0.1)
            parameters["b_conv3_3"] = nn_utils.cnn_utils.bias_convolution(256, 0.0)
            parameters["w_conv4_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 256, 512, 0.1)
            parameters["b_conv4_1"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
            parameters["w_conv4_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.1)
            parameters["b_conv4_2"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
            parameters["w_conv4_3"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.1)
            parameters["b_conv4_3"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
            parameters["w_conv5_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.1)
            parameters["b_conv5_1"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
            parameters["w_conv5_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.1)
            parameters["b_conv5_2"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
            parameters["w_conv5_3"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.1)
            parameters["b_conv5_3"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)

            parameters["w_fc7"] = nn_utils.cnn_utils.normal_weight_variable([4096, 4096], 0.1)
            parameters["b_fc7"] = nn_utils.cnn_utils.bias_variable([4096], 1.0)
            parameters["w_fc8"] = nn_utils.cnn_utils.normal_weight_variable([4096, 1000], 0.1)
            parameters["b_fc8"] = nn_utils.cnn_utils.bias_variable([1000], 1.0)
        else:
            parameters = shared_weight
        layers = {}
        if (input_layer is None):
            layers["image_input"] = tf.placeholder(tf.float32, shape = [\
                                       batch_size, self.image_height, self.image_width, self.image_channel])
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
        pool5_size = int(layers["pool5"].shape[1]) * int(layers["pool5"].shape[2]) * int(layers["pool5"].shape[3])
        layers["flat"] = tf.reshape(layers["pool5"], [-1, pool5_size])
        if (shared_weight is None):
            parameters["w_fc6"] = nn_utils.cnn_utils.normal_weight_variable([pool5_size, 4096], 0.1)
            parameters["b_fc6"] = nn_utils.cnn_utils.bias_variable([4096], 1.0)
        layers["fc6"] = tf.nn.bias_add(tf.matmul(layers["flat"], parameters["w_fc6"]), parameters["b_fc6"])
        layers["relu6"] = tf.nn.relu(layers["fc6"])
        layers["fc7"] = tf.nn.bias_add(tf.matmul(layers["relu6"], parameters["w_fc7"]), parameters["b_fc7"])
        layers["relu7"] = tf.nn.relu(layers["fc7"])
        layers["fc8"] = tf.nn.bias_add(tf.matmul(layers["relu7"], parameters["w_fc8"]), parameters["b_fc8"])
        layers["output"] = tf.nn.softmax(layers["fc8"])
        return [layers, parameters]

    def load_weight(self, sess, filename = None, weight_input = None, matching = None, display = False):
        """
            load pretrained weights into this model
            sess : tf.Session
            sess = session used for variable assignment
            filename : string
            filename = file name of pretrained weight file. if this is None, please provide weight_input
            weight_input : numpy.lib.npyio.NpzFile
            weight_input = pretrained weights. if this is None, please provide filename
            matching : dictionary
            matching = a mapping between names of pretrained weights and weights loaded in this model.
                       e.g. when matching['a'] = 'b', pretrained['b'] will be assigned to self.parameters['a']
            display : boolean
            display = indicator for if the process of loading should be displayed
        """
        if (matching is None):
            match = {"w_conv1_1": "conv1_1_W",
                     "b_conv1_1": "conv1_1_b",
                     "w_conv1_2": "conv1_2_W",
                     "b_conv1_2": "conv1_2_b",
                     "w_conv2_1": "conv2_1_W",
                     "b_conv2_1": "conv2_1_b",
                     "w_conv2_2": "conv2_2_W",
                     "b_conv2_2": "conv2_2_b",
                     "w_conv3_1": "conv3_1_W",
                     "b_conv3_1": "conv3_1_b",
                     "w_conv3_2": "conv3_2_W",
                     "b_conv3_2": "conv3_2_b",
                     "w_conv3_3": "conv3_3_W",
                     "b_conv3_3": "conv3_3_b",
                     "w_conv4_1": "conv4_1_W",
                     "b_conv4_1": "conv4_1_b",
                     "w_conv4_2": "conv4_2_W",
                     "b_conv4_2": "conv4_2_b",
                     "w_conv4_3": "conv4_3_W",
                     "b_conv4_3": "conv4_3_b",
                     "w_conv5_1": "conv5_1_W",
                     "b_conv5_1": "conv5_1_b",
                     "w_conv5_2": "conv5_2_W",
                     "b_conv5_2": "conv5_2_b",
                     "w_conv5_3": "conv5_3_W",
                     "b_conv5_3": "conv5_3_b",
                     "w_fc6": "fc6_W",
                     "b_fc6": "fc6_b",
                     "w_fc7": "fc7_W",
                     "b_fc7": "fc7_b",
                     "w_fc8": "fc8_W",
                     "b_fc8": "fc8_b"
                    }
        else:
            match = matching
        if (weight_input is None):
            weight_loaded = np.load(filename)
        else:
            weight_loaded = weight_input
        for para_name in match:
            if (display):
                print para_name, match[para_name]
            sess.run(self.parameters[para_name].assign(weight_loaded[match[para_name]]))


