from __future__ import division
import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils
from my_vgg.vgg16_model import VGG16_model
import scipy.io

class FCN_model:
    """
        My Implementation for FCN-8s
    """
    def __init__(self, num_class, convert_from_vgg = None, drop_out_prob = 0.5, img_height = None, img_width = None, img_channel = None, training_batch_size = None, test_batch_size = None, sess = None, learning_rate = 1e-4, weight_decay_coefficient = 5e-4, vlfeat_model_path = None):
        """
            img_height : int
            img_height = height of images
            num_class : int
            num_class = number of output class
            convert_from_vgg : VGG16_model
            convert_from_vgg = the VGG16 model which will be converted to FCN.
                               when an VGG16 model is converted to FCN, they will share the same weight object,
                               i.e. any modification in these weights will appear in both models
            training_batch_size : int
            training_batch_size = batch size of training batch
            test_batch_size : int
            test_batch_size = batch size of test batch
            drop_out_prob : float
            drop_out_prob = probability for drop out
            sess : tensorflow.Session
            sess = session used for coverting parameters from VGG, or loading pretrained model
            learning_rate : float
            learning_rate = when a training network is constructed, this will be the learning rate
            weight_decay_coefficient : float
            weight_decay_coefficient = coefficient for weight decay
            vlfeat_model_path : string
            vlfeat_model_path = when this is not None, an vlfeat pretrained model in this path will be loaded
        """
        self.num_class = num_class
        self.parameters = self.empty_parameters(num_class)
        if (vlfeat_model_path is not None):
            _ = self.load_vlfeat_pretrained_model(vlfeat_model_path, sess)
        self.weight_decay_loss = self.weight_decay(weight_decay_coefficient)
        self.base_lr = learning_rate
        if ((convert_from_vgg is not None) and (sess is not None)):
            self.parameters = self.convert_VGG(convert_from_vgg, num_class, sess)
        if (training_batch_size is not None):
            self.train_net = self.build_computation_graph(self.parameters, training_batch_size, num_class, drop_out_prob, img_height, img_width, img_channel)
            self.train_net["ground_truth"] = tf.placeholder(tf.int32, shape = [training_batch_size, img_height, img_width])
            self.train_cross_entropy = self.cross_entropy(self.train_net["score_output"], self.train_net["ground_truth"])
            self.train_loss = self.train_cross_entropy + self.weight_decay_loss
            self.train_update = tf.train.AdamOptimizer(learning_rate).minimize(self.train_loss)
        if (test_batch_size is not None):
            self.test_net = self.build_computation_graph(self.parameters, test_batch_size, num_class, drop_out_prob, img_height, img_width, img_channel, train_net = False)
            self.test_net["ground_truth"] = tf.placeholder(tf.int32, shape = [test_batch_size, img_height, img_width])
            self.test_cross_entropy = self.cross_entropy(self.test_net["score_output"], self.test_net["ground_truth"])

    def training_minimizer(self, optimizer, loss, lr_mult = None):
        """
            construct a customized loss function optimizer with different learning rate for different variables
            optimizer : tf.train.Optimizer
            optimizer = the optimizer used for minimizing the loss function
            loss : tensorflow.python.framework.ops.Tensor
            loss = loss function which should be minimized
            lr_mult : dictionary
            lr_mult = a collection of learning rate coefficient. 
                      i.e. the learning rate for variable x, should be self.base_lr * lr_mult[x]
            ------------------------------------------------------------------------------------
            return minimizer
            minimizer : tensorflow.python.framework.ops.Tensor
            minimizer = the customized minimizer
        """
        if (lr_mult is None):
            lr_mult = {}
            lr_mult["w_conv1_1"] = 1.0
            lr_mult["b_conv1_1"] = 2.0
            lr_mult["w_conv1_2"] = 1.0
            lr_mult["b_conv1_2"] = 2.0
            lr_mult["w_conv2_1"] = 1.0
            lr_mult["b_conv2_1"] = 2.0
            lr_mult["w_conv2_2"] = 1.0
            lr_mult["b_conv2_2"] = 2.0
            lr_mult["w_conv3_1"] = 1.0
            lr_mult["b_conv3_1"] = 2.0
            lr_mult["w_conv3_2"] = 1.0
            lr_mult["b_conv3_2"] = 2.0
            lr_mult["w_conv3_3"] = 1.0
            lr_mult["b_conv3_3"] = 2.0
            lr_mult["w_conv4_1"] = 1.0
            lr_mult["b_conv4_1"] = 2.0
            lr_mult["w_conv4_2"] = 1.0
            lr_mult["b_conv4_2"] = 2.0
            lr_mult["w_conv4_3"] = 1.0
            lr_mult["b_conv4_3"] = 2.0
            lr_mult["w_conv5_1"] = 1.0
            lr_mult["b_conv5_1"] = 2.0
            lr_mult["w_conv5_2"] = 1.0
            lr_mult["b_conv5_2"] = 2.0
            lr_mult["w_conv5_3"] = 1.0
            lr_mult["b_conv5_3"] = 2.0
            lr_mult["w_conv6"] = 1.0
            lr_mult["b_conv6"] = 2.0
            lr_mult["w_conv7"] = 1.0
            lr_mult["b_conv7"] = 2.0
            lr_mult["w_score_up1"] = 1.0
            lr_mult["b_score_up1"] = 2.0
            lr_mult["w_score_up2"] = 1.0
            lr_mult["w_score_pool4"] = 1.0
            lr_mult["b_score_pool4"] = 2.0
            lr_mult["w_score_up4"] = 1.0
            lr_mult["w_score_pool3"] = 1.0
            lr_mult["b_score_pool3"] = 2.0
            lr_mult["w_score_output"] = 1.0
        grads_and_vars = optimizer.compute_gradients(loss)
        grads_and_vars_dict = {}
        for g_v_pair in grads_and_vars:
            grads_and_vars_dict[g_v_pair[1]] = g_v_pair[0]
        customized_g_v = []
        for p in self.parameters:
            customized_g_v.append((grads_and_vars_dict[self.parameters[p]] * lr_mult[p], self.parameters[p]))
        return optimizer.apply_gradients(customized_g_v)

    def build_computation_graph(self, parameters, batch_size, num_class, drop_out_prob = 0.5, img_height = None, img_width = None, img_channel = None, input_layer = None, train_net = True):
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
            train_net : boolean
            train_net = when this is true, there will be no drop out
            ------------------------------------------------
            return layers
            layers : dictionary
            layers = collection of tensors for each layers, indexed by name
        """
        layers = {}
        if (input_layer is None):
            layers["image_input"] = tf.placeholder(tf.float32, shape = [\
                                       batch_size, img_height, img_width, img_channel])
        else:
            layers["image_input"] = input_layer
        layers["img_processed"] = layers["image_input"] - parameters["mean_rgb"]
        layers["conv1_1"] = tf.nn.bias_add(tf.nn.conv2d(layers["img_processed"],
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
        conv6_result = "relu6"
        if (train_net):
            layers["dropout6"] = tf.nn.dropout(layers["relu6"], drop_out_prob)
            conv6_result = "dropout6"
        layers["conv7"] = tf.nn.bias_add(tf.nn.conv2d(layers[conv6_result], parameters["w_conv7"], strides = [1, 1, 1, 1], \
                                         padding = 'SAME'), parameters["b_conv7"])
        layers["relu7"] = tf.nn.relu(layers["conv7"])
        conv7_result = "relu7"
        if (train_net):
            layers["dropout7"] = tf.nn.dropout(layers["relu7"], drop_out_prob)
            conv7_result = "dropout7"
        layers["score_up1"] = tf.nn.bias_add(tf.nn.conv2d(layers[conv7_result], parameters["w_score_up1"], strides = [1, 1, 1, 1], \
                                          padding = 'SAME'), parameters["b_score_up1"])
        pool4_shape = [int(layers["pool4"].get_shape()[0]), int(layers["pool4"].get_shape()[1]), \
                       int(layers["pool4"].get_shape()[2]), int(layers["pool4"].get_shape()[3])]
        layers["score_up2"] = tf.nn.bias_add(tf.nn.conv2d_transpose(layers["score_up1"], parameters["w_score_up2"], \
                                                     output_shape = [pool4_shape[0], pool4_shape[1], pool4_shape[2], num_class], \
                                                     strides = [1, 2, 2, 1], padding = 'SAME'), parameters["b_score_up2"])
        layers["pool4_scale"] = layers["pool4"]
        layers["score_pool4"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool4_scale"], parameters["w_score_pool4"], \
                                               strides = [1, 1, 1, 1], padding = 'SAME'), parameters["b_score_pool4"])
        layers["fuse_pool4"] = tf.add(layers["score_up2"], layers["score_pool4"])
        pool3_shape = [int(layers["pool3"].get_shape()[0]), int(layers["pool3"].get_shape()[1]), \
                       int(layers["pool3"].get_shape()[2]), int(layers["pool3"].get_shape()[3])]
        layers["score_up4"] = tf.nn.conv2d_transpose(layers["fuse_pool4"], parameters["w_score_up4"], \
                                                     output_shape = [pool3_shape[0], pool3_shape[1], pool3_shape[2], num_class], \
                                                     strides = [1, 2, 2, 1], padding = 'SAME')
        layers["pool3_scale"] = layers["pool3"]
        layers["score_pool3"] = tf.nn.bias_add(tf.nn.conv2d(layers["pool3_scale"], parameters["w_score_pool3"], \
                                               strides = [1, 1, 1, 1], padding = 'SAME'), parameters["b_score_pool3"])
        layers["fuse_pool3"] = tf.add(layers["score_pool3"], layers["score_up4"])
        layers["score_output"] = tf.nn.conv2d_transpose(layers["fuse_pool3"], parameters["w_score_output"], \
                                                        output_shape = [batch_size, img_height, img_width, num_class], \
                                                        strides = [1, 8, 8, 1], padding = 'SAME')
        return layers

    def infer_an_image(self, image, sess, color):
        """
            infer for an image
            image : numpy.ndarray
            image = image which will be infered, shape (image_height, image_width, image_channel = 3)
            sess : tensorflow.Session
            sess = tensorflow session, which will be used for infering
            color : list
            colot = a list of color, color[i] is color for class i, which is RGB
            ----------------------------------------------------------------
            return [score, segmentation, visualization]
            score : numpy.ndarray
            score = score output for each class, shape (image_height, image_width, num_class)
            segmentation : numpy.ndarray
            segmentation = segmentation result, label for each pixel, shape (image_height, image_width)
            visualization : numpy.ndarray
            visualization = visualization of segmentation, RGB coloe image in the same color map as the one used in dataset, shape (image_height, image_width, 3)
        """
        num_class = int(self.parameters["w_score_output"].get_shape()[3])
        img_height = image.shape[0]
        img_width = image.shape[1]
        img_channel = image.shape[2]
        layers = self.build_computation_graph(self.parameters, 1, num_class, img_height = img_height, img_width = img_width, img_channel = img_channel, train_net = False)
        layers["output"] = tf.argmax(layers["score_output"], axis = 3)
        score_raw, segmentation_raw = sess.run([layers["score_output"], layers["output"]], feed_dict = {layers["image_input"] : np.reshape(image, [1, img_height, img_width, img_channel])})
        if (color is None):
            return [score_raw[0], segmentation_raw[0], None]
        else:
            visualization = np.zeros([img_height, img_width, 3], np.int)
            for i in range(visualization.shape[0]):
                for j in range(visualization.shape[1]):
                    label = segmentation_raw[0, i, j]
                    visualization[i, j, 0] = color[label][0]
                    visualization[i, j, 1] = color[label][1]
                    visualization[i, j, 2] = color[label][2]
            return [score_raw[0], segmentation_raw[0], visualization]

    def convert_VGG(self, vgg_model, num_class, sess):
        """
            convert weights from VGG model to FCN
            vgg_model : VGG16_model
            vgg_model = the vgg model which will be converted
            num_class : int
            num_class = number of output classes
            sess : tensorflow.Session
            sess = session used for variable assignment
            ---------------------------------------------------
            return parameters
            parameters : dictionary
            parameters = collection of parameters used in this architecture, indexed by name
        """
        shared_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', \
                         'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
        extended_layers = ['score_up1', 'score_up2', 'score_pool4', 'score_up4', \
                           'score_pool3', 'score_output']
        for layer in shared_layers:
            sess.run(self.parameters['w_' + layer].assign(vgg_model.parameters['w_' + layer]))
            sess.run(self.parameters['b_' + layer].assign(vgg_model.parameters['b_' + layer]))
        sess.run(self.parameters["w_conv6"].assign(tf.reshape(vgg_model.parameters["w_fc6"], [7, 7, 512, 4096])))
        sess.run(self.parameters["b_conv6"].assign(vgg_model.parameters["b_fc6"]))
        sess.run(self.parameters["w_conv7"].assign(tf.reshape(vgg_model.parameters["w_fc7"], [1, 1, 4096, 4096])))
        sess.run(self.parameters["b_conv7"].assign(vgg_model.parameters["b_fc7"]))
        sess.run(self.parameters["w_score_up2"].assign(self.bilinear_filter(4, 4, num_class, num_class)))
        sess.run(self.parameters["w_score_up4"].assign(self.bilinear_filter(4, 4, num_class, num_class)))
        sess.run(self.parameters["w_score_output"].assign(self.bilinear_filter(16, 16, num_class, num_class)))

    def convert_VGG_npz(self, vgg_npz_path, sess):
        """
            load weights of VGG from npz file
            https://github.com/tensorflow/models/tree/master/slim
            vgg_npz_path : string
            vgg_npz_path = path to the npz file
            sess : tensorflow.Session
            sess = session used for variable assignment
        """
        shared_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', \
                         'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
        self.parameters["mean_rgb"][:] = [123.68, 116.78, 103.94]
        raw = np.load(vgg_npz_path)
        for layer in shared_layers:
            sess.run(self.parameters['w_' + layer].assign(raw['w_' + layer]))
            sess.run(self.parameters['b_' + layer].assign(raw['b_' + layer]))
        sess.run(self.parameters['w_conv6'].assign(raw['w_fc6']))
        sess.run(self.parameters['b_conv6'].assign(raw['b_fc6']))
        sess.run(self.parameters['w_conv7'].assign(raw['w_fc7']))
        sess.run(self.parameters['b_conv7'].assign(raw['b_fc7']))
        sess.run(self.parameters["w_score_up2"].assign(self.bilinear_filter(4, 4, self.num_class, self.num_class)))
        sess.run(self.parameters["w_score_up4"].assign(self.bilinear_filter(4, 4, self.num_class, self.num_class)))
        sess.run(self.parameters["w_score_output"].assign(self.bilinear_filter(16, 16, self.num_class, self.num_class)))

    def empty_parameters(self, num_class):
        """
            construct parameters
            num_class : int
            num_class = number of class
            --------------------------------------------------
            return parameters
            parameters : dictionary
            parameters = collection of extended parameters, indexed by name
        """
        parameters = {}
        parameters["mean_rgb"] = np.zeros([3])
        parameters["w_conv1_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 3, 64, 0.0001)
        parameters["b_conv1_1"] = nn_utils.cnn_utils.bias_convolution(64, 0.0)
        parameters["w_conv1_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 64, 64, 0.0001)
        parameters["b_conv1_2"] = nn_utils.cnn_utils.bias_convolution(64, 0.0)
        parameters["w_conv2_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 64, 128, 0.0001)
        parameters["b_conv2_1"] = nn_utils.cnn_utils.bias_convolution(128, 0.0)
        parameters["w_conv2_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 128, 128, 0.0001)
        parameters["b_conv2_2"] = nn_utils.cnn_utils.bias_convolution(128, 0.0)
        parameters["w_conv3_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 128, 256, 0.0001)
        parameters["b_conv3_1"] = nn_utils.cnn_utils.bias_convolution(256, 0.0)
        parameters["w_conv3_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 256, 256, 0.0001)
        parameters["b_conv3_2"] = nn_utils.cnn_utils.bias_convolution(256, 0.0)
        parameters["w_conv3_3"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 256, 256, 0.0001)
        parameters["b_conv3_3"] = nn_utils.cnn_utils.bias_convolution(256, 0.0)
        parameters["w_conv4_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 256, 512, 0.0001)
        parameters["b_conv4_1"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
        parameters["w_conv4_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.0001)
        parameters["b_conv4_2"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
        parameters["w_conv4_3"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.0001)
        parameters["b_conv4_3"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
        parameters["w_conv5_1"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.0001)
        parameters["b_conv5_1"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
        parameters["w_conv5_2"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.0001)
        parameters["b_conv5_2"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
        parameters["w_conv5_3"] = nn_utils.cnn_utils.weight_convolution_normal([3, 3], 512, 512, 0.0001)
        parameters["b_conv5_3"] = nn_utils.cnn_utils.bias_convolution(512, 0.0)
        parameters["w_conv6"] = nn_utils.cnn_utils.weight_convolution_normal([7, 7], 512, 4096, 0.0001)
        parameters["b_conv6"] = nn_utils.cnn_utils.bias_convolution(4096, 0.0)
        parameters["w_conv7"] = nn_utils.cnn_utils.weight_convolution_normal([1, 1], 4096, 4096, 0.0001)
        parameters["b_conv7"] = nn_utils.cnn_utils.bias_convolution(4096, 0.0)
        ext = self.extend_parameters(num_class)
        for layer in ext:
            parameters[layer] = ext[layer]
        return parameters

    def extend_parameters(self, num_class):
        """
            construct parameters for layers in FCN but not in VGG
            num_class : int
            num_class = number of output classes
            --------------------------------------------------
            return parameters
            parameters : dictionary
            parameters = collection of extended parameters, indexed by name
        """
        parameters = {}
        parameters["w_score_up1"] = nn_utils.cnn_utils.weight_convolution_normal([1, 1], 4096, num_class, 0.1)
        parameters["b_score_up1"] = nn_utils.cnn_utils.bias_convolution(num_class, 0.0)
        parameters["w_score_up2"] = nn_utils.cnn_utils.weight_deconvolution_normal([4, 4], num_class, num_class, 0.1)
        parameters["b_score_up2"] = nn_utils.cnn_utils.bias_convolution(num_class, 0.0)
        parameters["w_score_pool4"] = nn_utils.cnn_utils.weight_convolution_normal([1, 1], 512, num_class, 0.1)
        parameters["b_score_pool4"] = nn_utils.cnn_utils.bias_convolution(num_class, 0.0)
        parameters["w_score_up4"] = nn_utils.cnn_utils.weight_deconvolution_normal([4, 4], num_class, num_class, 0.1)
        parameters["w_score_pool3"] = nn_utils.cnn_utils.weight_convolution_normal([1, 1], 256, num_class, 0.1)
        parameters["b_score_pool3"] = nn_utils.cnn_utils.bias_convolution(num_class, 0.0)
        parameters["w_score_output"] = nn_utils.cnn_utils.weight_deconvolution_normal([16, 16], num_class, num_class, 0.1)
        return parameters

    def bilinear_filter(self, filter_height, filter_width, channel_input, channel_output):
        """
            get a filter of bilinear
            filter_height : int
            filter_height = height of the filter
            filter_width : int
            filter_width = width of the filter
            channel_input : int
            channel_input = number of input channels
            channel_output : int
            channel_output = number of output channels
            ----------------------------------------------------
            return weight
            weight : np.ndarray
            weight = the filter for bilinear, shape (filter_height, filter_width, channel_output, channel_input)
        """
        weight = np.zeros([filter_height, filter_width, channel_output, channel_input], np.float)
        bilinear = np.zeros([filter_height, filter_width], np.float)
        factor = [(filter_height + 1) // 2, (filter_width + 1) // 2]
        center = []
        if (filter_height % 2 == 1):
            center.append(factor[0] - 1)
        else:
            center.append(factor[0] - 0.5)
        if (filter_width % 2 == 1):
            center.append(factor[1] - 1)
        else:
            center.append(factor[1] - 0.5)
        for i in range(bilinear.shape[0]):
            for j in range(bilinear.shape[1]):
                f0 = 1 - abs(i - center[0]) / factor[0]
                f1 = 1 - abs(j - center[1]) / factor[1]
                bilinear[i, j] = f0 * f1
        for i in range(weight.shape[2]):
            for j in range(weight.shape[3]):
                weight[:, :, i, j] = bilinear
        return weight

    def cross_entropy(self, logit_layer, label):
        """
            construct a computation node of loss
            logit_layer : tensorflow.python.framework.ops.Tensor
            logit_layer = the last layer of the architecture before softmax. shape [batch_size, img_height, img_width, num_class]
            label : tensorflow.python.framework.ops.Tensor
            label = placeholder for the ground truth. shape [batch_size, img_height, img_width]
            -----------------------------------------------------------
            return loss
            loss : tensorflow.python.framework.ops.Tensor
            loss = a computation node for loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit_layer, labels = label))

    def weight_decay(self, weight_decay_coefficient, weights_list = None):
        """
            weight decay
            weight_decay_coefficient : float
            weight_decay_coefficient = coefficient for weight decay
            weights_list : list
            weights_list = when this is not None, return a weight decay from this list
            -----------------------------------------------------------------
            return wd
            wd : tensorflow.python.framework.ops.Tensor
            wd = weight decay
        """
        if (weights_list is None):
            wd_list = ["w_conv1_1", "w_conv1_2", "w_conv2_1", "w_conv2_2", "w_conv3_1", "w_conv3_2", "w_conv3_3",\
                       "w_conv4_1", "w_conv4_2", "w_conv4_3", "w_conv5_1", "w_conv5_2", "w_conv5_3", "w_conv6",\
                       "w_conv7", "w_score_up1", "w_score_up2", "w_score_pool4", "w_score_up4", "w_score_pool3",\
                       "w_score_output"]
            weights_list = []
            for layer in wd_list:
                weights_list.append(self.parameters[layer])
        for weight in weights_list:
            tf.add_to_collection("weight_decay_regularization",\
                                  tf.multiply(tf.nn.l2_loss(weight), weight_decay_coefficient))
        return tf.add_n(tf.get_collection("weight_decay_regularization"))

    def save_weights_to_npz(self, path_to_file, sess):
        """
            save weights to npz file
            path_to_file : string
            path_to_file = path to the saved file
            sess : tf.Session
            sess = tensorflow session, used for computing weights
        """
        w_conv1_1 = sess.run(self.parameters["w_conv1_1"])
        b_conv1_1 = sess.run(self.parameters["b_conv1_1"])
        w_conv1_2 = sess.run(self.parameters["w_conv1_2"])
        b_conv1_2 = sess.run(self.parameters["b_conv1_2"])
        w_conv2_1 = sess.run(self.parameters["w_conv2_1"])
        b_conv2_1 = sess.run(self.parameters["b_conv2_1"])
        w_conv2_2 = sess.run(self.parameters["w_conv2_2"])
        b_conv2_2 = sess.run(self.parameters["b_conv2_2"])
        w_conv3_1 = sess.run(self.parameters["w_conv3_1"])
        b_conv3_1 = sess.run(self.parameters["b_conv3_1"])
        w_conv3_2 = sess.run(self.parameters["w_conv3_2"])
        b_conv3_2 = sess.run(self.parameters["b_conv3_2"])
        w_conv3_3 = sess.run(self.parameters["w_conv3_3"])
        b_conv3_3 = sess.run(self.parameters["b_conv3_3"])
        w_conv4_1 = sess.run(self.parameters["w_conv4_1"])
        b_conv4_1 = sess.run(self.parameters["b_conv4_1"])
        w_conv4_2 = sess.run(self.parameters["w_conv4_2"])
        b_conv4_2 = sess.run(self.parameters["b_conv4_2"])
        w_conv4_3 = sess.run(self.parameters["w_conv4_3"])
        b_conv4_3 = sess.run(self.parameters["b_conv4_3"])
        w_conv5_1 = sess.run(self.parameters["w_conv5_1"])
        b_conv5_1 = sess.run(self.parameters["b_conv5_1"])
        w_conv5_2 = sess.run(self.parameters["w_conv5_2"])
        b_conv5_2 = sess.run(self.parameters["b_conv5_2"])
        w_conv5_3 = sess.run(self.parameters["w_conv5_3"])
        b_conv5_3 = sess.run(self.parameters["b_conv5_3"])
        w_conv6 = sess.run(self.parameters["w_conv6"])
        b_conv6 = sess.run(self.parameters["b_conv6"])
        w_conv7 = sess.run(self.parameters["w_conv7"])
        b_conv7 = sess.run(self.parameters["b_conv7"])
        w_score_up1 = sess.run(self.parameters["w_score_up1"])
        b_score_up1 = sess.run(self.parameters["b_score_up1"])
        w_score_up2 = sess.run(self.parameters["w_score_up2"])
        w_score_pool4 = sess.run(self.parameters["w_score_pool4"])
        b_score_pool4 = sess.run(self.parameters["b_score_pool4"])
        w_score_up4 = sess.run(self.parameters["w_score_up4"])
        w_score_pool3 = sess.run(self.parameters["w_score_pool3"])
        b_score_pool3 = sess.run(self.parameters["b_score_pool3"])
        w_score_output = sess.run(self.parameters["w_score_output"])
        np.savez(path_to_file, \
                 w_conv1_1 = w_conv1_1, b_conv1_1 = b_conv1_1, w_conv1_2 = w_conv1_2, b_conv1_2 = b_conv1_2,\
                 w_conv2_1 = w_conv2_1, b_conv2_1 = b_conv2_1, w_conv2_2 = w_conv2_2, b_conv2_2 = b_conv2_2,\
                 w_conv3_1 = w_conv3_1, b_conv3_1 = b_conv3_1, w_conv3_2 = w_conv3_2, b_conv3_2 = b_conv3_2, w_conv3_3 = w_conv3_3, b_conv3_3 = b_conv3_3,\
                 w_conv4_1 = w_conv4_1, b_conv4_1 = b_conv4_1, w_conv4_2 = w_conv4_2, b_conv4_2 = b_conv4_2, w_conv4_3 = w_conv4_3, b_conv4_3 = b_conv4_3,\
                 w_conv5_1 = w_conv5_1, b_conv5_1 = b_conv5_1, w_conv5_2 = w_conv5_2, b_conv5_2 = b_conv5_2, w_conv5_3 = w_conv5_3, b_conv5_3 = b_conv5_3,\
                 w_conv6 = w_conv6, b_conv6 = b_conv6, w_conv7 = w_conv7, b_conv7 = b_conv7,\
                 w_score_up1 = w_score_up1, b_score_up1 = b_score_up1, w_score_up2 = w_score_up2, \
                 w_score_pool4 = w_score_pool4, b_score_pool4 = b_score_pool4,\
                 w_score_up4 = w_score_up4, w_score_pool3 = w_score_pool3, b_score_pool3 = b_score_pool3,\
                 w_score_output = w_score_output, mean_rgb = self.parameters["mean_rgb"])

    def load_weights_from_npz(self, path_to_npz, sess, save_to_this = True, fine_tune_to_new_label = False):
        """
            load weights from npz file
            path_to_npz : string
            path_to_npz = path to npz file
            sess : tensorflow.Session
            sess = tensorflow session used for variable assignment
            save_to_this : boolean
            save_to_this = when this is true, the parameter is saved as parameter of this object
            fine_tune_to_new_label : boolean
            fine_tune_to_new_label = when this is True, 
                                     class relayed layers (score layers) will not be loaded
            --------------------------------------------------
            return parameters
            parameters : dictionary
            parameters = collection of extended parameters, indexed by name
        """
        weight_loaded = np.load(path_to_npz)
        if (fine_tune_to_new_label):
            num_class = self.num_class
        else:
            num_class = weight_loaded['w_score_output'].shape[3]
        if (save_to_this):
            parameters = self.parameters
        else:
            parameters = self.empty_parameters(num_class)
        para_list = []
        for layer in parameters:
            if (layer != 'mean_rgb'):
                para_list.append(parameters[layer])
        sess.run(tf.variables_initializer(var_list=para_list))
        if (fine_tune_to_new_label):
            layers_class_related = ['score_up1', 'score_up2', 'score_pool4', 'score_up4', 'score_pool3', 'score_output']
        for layer in weight_loaded:
            if (layer != 'mean_rgb'):
                if ((not fine_tune_to_new_label) or (layer[2:] not in layers_class_related)):
                    sess.run(parameters[layer].assign(weight_loaded[layer]))
                elif (layer == 'w_score_up2'):
                    sess.run(parameters[layer].assign(self.bilinear_filter(4, 4, self.num_class, self.num_class)))
                elif (layer == 'w_score_up4'):
                    sess.run(parameters[layer].assign(self.bilinear_filter(4, 4, self.num_class, self.num_class)))
                elif (layer == 'w_score_output'):
                    sess.run(parameters[layer].assign(self.bilinear_filter(16, 16, self.num_class, self.num_class)))
        parameters["mean_rgb"][:] = weight_loaded["mean_rgb"]
        return parameters

    def load_vlfeat_pretrained_model(self, model_path, sess, save_to_this = True):
        """
            load vlfeat pretrained model
            model_path : string
            model_path = path to the pretrained model
            sess : tensorflow.Session
            sess = tensorflow session used for variable assignment
            save_to_this : boolean
            save_to_this = when this is True
            ---------------------------------------------------
            return parameters
            parameters : dictionary
            parameters = collection of parameters
        """
        mat_raw = scipy.io.loadmat(model_path)
        param_raw = mat_raw['params']
        num_class = param_raw[0, 39][1].shape[2]
        if (save_to_this):
            parameters = self.parameters
        else:
            parameters = self.empty_parameters(num_class)
        self.mean_rgb = mat_raw['meta'][0, 0][2][0][0][2][0, 0]
        self.parameters["mean_rgb"][:] = self.mean_rgb
        sess.run(parameters["w_conv1_1"].assign(param_raw[0, 0][1]))
        sess.run(parameters["b_conv1_1"].assign(param_raw[0, 1][1][:, 0]))
        sess.run(parameters["b_conv1_2"].assign(param_raw[0, 3][1][:, 0]))
        sess.run(parameters["w_conv2_1"].assign(param_raw[0, 4][1]))
        sess.run(parameters["b_conv2_1"].assign(param_raw[0, 5][1][:, 0]))
        sess.run(parameters["w_conv2_2"].assign(param_raw[0, 6][1]))
        sess.run(parameters["b_conv2_2"].assign(param_raw[0, 7][1][:, 0]))
        sess.run(parameters["w_conv3_1"].assign(param_raw[0, 8][1]))
        sess.run(parameters["b_conv3_1"].assign(param_raw[0, 9][1][:, 0]))
        sess.run(parameters["w_conv3_2"].assign(param_raw[0, 10][1]))
        sess.run(parameters["b_conv3_2"].assign(param_raw[0, 11][1][:, 0]))
        sess.run(parameters["w_conv3_3"].assign(param_raw[0, 12][1]))
        sess.run(parameters["b_conv3_3"].assign(param_raw[0, 13][1][:, 0]))
        sess.run(parameters["w_conv4_1"].assign(param_raw[0, 14][1]))
        sess.run(parameters["b_conv4_1"].assign(param_raw[0, 15][1][:, 0]))
        sess.run(parameters["w_conv4_2"].assign(param_raw[0, 16][1]))
        sess.run(parameters["b_conv4_2"].assign(param_raw[0, 17][1][:, 0]))
        sess.run(parameters["w_conv4_3"].assign(param_raw[0, 18][1]))
        sess.run(parameters["b_conv4_3"].assign(param_raw[0, 19][1][:, 0]))
        sess.run(parameters["w_conv5_1"].assign(param_raw[0, 20][1]))
        sess.run(parameters["b_conv5_1"].assign(param_raw[0, 21][1][:, 0]))
        sess.run(parameters["w_conv5_2"].assign(param_raw[0, 22][1]))
        sess.run(parameters["b_conv5_2"].assign(param_raw[0, 23][1][:, 0]))
        sess.run(parameters["w_conv5_3"].assign(param_raw[0, 24][1]))
        sess.run(parameters["b_conv5_3"].assign(param_raw[0, 25][1][:, 0]))
        sess.run(parameters["w_conv6"].assign(param_raw[0, 26][1]))
        sess.run(parameters["b_conv6"].assign(param_raw[0, 27][1][:, 0]))
        sess.run(parameters["w_conv7"].assign(param_raw[0, 28][1]))
        sess.run(parameters["b_conv7"].assign(param_raw[0, 29][1][:, 0]))
        sess.run(parameters["w_score_up1"].assign(param_raw[0, 30][1]))
        sess.run(parameters["b_score_up1"].assign(param_raw[0, 31][1][:, 0]))
        sess.run(parameters["w_score_up2"].assign(param_raw[0, 32][1]))
        sess.run(parameters["b_score_up2"].assign(param_raw[0, 33][1][:, 0]))
        sess.run(parameters["w_score_pool4"].assign(param_raw[0, 34][1]))
        sess.run(parameters["b_score_pool4"].assign(param_raw[0, 35][1][:, 0]))
        sess.run(parameters["w_score_up4"].assign(param_raw[0, 36][1]))
        sess.run(parameters["w_score_pool3"].assign(param_raw[0, 37][1]))
        sess.run(parameters["b_score_pool3"].assign(param_raw[0, 38][1][:, 0]))
        sess.run(parameters["w_score_output"].assign(param_raw[0, 39][1]))
        return parameters
