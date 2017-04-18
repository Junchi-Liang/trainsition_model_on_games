import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils
import data_utils.data_read_util
import data_utils.img_process_util

class Multi_Step_net:
    """
        Neural Networks with multi-step prediction
    """
    def __init__(self, stacked_img_num = 4, image_height = 84, image_width =\
                 84, num_actions = 18, num_channels = 1, training_batch_size =\
                32, test_batch_size = None, step_prediction = 7):
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
            step_prediction : int
            step_prediction - number of steps for prediction in training
        """
        self.img_height = image_height
        self.img_width = image_width
        self.num_act = num_actions
        self.num_stack = stacked_img_num
        self.num_ch = num_channels
        self.num_step = step_prediction

        self.step_train_net_layer = []
        self.step_train_net_param = []
        for step_cur in range(step_prediction):
            if (step_cur == 0):
                layer_cur, param_cur = self.construct_network_computational_graph(batch_size = training_batch_size)
            else:
                concat_layer = tf.concat([self.step_train_net_layer[step_cur - 1]["deconv3"], \
                                   self.step_train_net_layer[step_cur - 1]["img_stacked_input_placeholder"][:, :, :, 0 : self.num_stack - 1]], axis = 3)
                layer_cur, param_cur = self.construct_network_computational_graph(batch_size = training_batch_size, input_layer = concat_layer)
            self.step_train_net_layer.append(layer_cur)
            self.step_train_net_param.append(param_cur)

        self.net_predict_layer, _ = self.construct_network_computational_graph(batch_size = 1, params_shared = self.step_train_net_param[0])
        if  (not(test_batch_size is None)):
            self.net_test = self.construct_network_computational_graph(batch_size = test_batch_size, params_shared = self.step_train_net_param[0])
        self.train_multi_step_loss = []
        self.train_multi_step_loss.append(self.mean_square_loss(self.step_train_net_layer[0]["deconv3"], self.step_train_net_layer[0]["frame_next_img_placeholder"]))
        for step_cur in range(1, step_prediction):
            self.train_multi_step_loss.append(self.train_multi_step_loss[step_cur - 1] + self.mean_square_loss(self.step_train_net_layer[step_cur]["deconv3"],\
                                            self.step_train_net_layer[step_cur]["frame_next_img_placeholder"]))
        self.train_step = [tf.train.RMSPropOptimizer(1e-4, momentum = 0.9).minimize(loss) for loss in self.train_multi_step_loss]

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
        params = {}
        if (params_shared is None):
            params["w_conv1"] = nn_utils.cnn_utils.weight_convolution_normal([6, 6], self.num_stack, 64, 0.1)
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
        if (input_layer is None):
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

    def train_iteration(self, accumulated_step, tf_sess = None, batch = None, arg_for_batch = None, arg_for_normalization = None, display = True):
        """
        """
        if (batch is not None):
            batch_input = batch
        else:
            episode_list, action_file_dict, reward_file_dict, stacked_img_num, frame_stride, batch_size,  prediction_step, img_dict, dataset, selected_index = arg_for_batch
            batch_input, _ = data_utils.data_read_util.minibatch_multi_step(episode_list, action_file_dict, reward_file_dict, stacked_img_num, frame_stride, batch_size,  prediction_step, img_dict, dataset, selected_index)
        stacked_img_tensor, reward_tensor, action_tensor, ground_truth_tensor = batch_input
        feed_for_net = {}
        if (arg_for_normalization is None):
            feed_for_net[self.step_train_net_layer[0]["img_stacked_input_placeholder"]] = stacked_img_tensor
        else:
            normalized_img = data_utils.img_process_util.img_normalize(stacked_img_tensor, arg_for_normalization)
            feed_for_net[self.step_train_net_layer[0]["img_stacked_input_placeholder"]] = normalized_img
        for i in range(self.num_step):
            feed_for_net[self.step_train_net_layer[i]["action_input_placeholder"]] = data_utils.data_read_util.one_hot_action(action_tensor[:, i], self.num_act)
            if (arg_for_normalization is None):
                feed_for_net[self.step_train_net_layer[i]["frame_next_img_placeholder"]] = ground_truth_tensor[:, :, :, i:i+1]
            else:
                normalized_ground_truth = data_utils.img_process_util.img_normalize(ground_truth_tensor[:, :, :, i:i+1], arg_for_normalization)
                feed_for_net[self.step_train_net_layer[i]["frame_next_img_placeholder"]] = normalized_ground_truth
        self.train_step[accumulated_step - 1].run(feed_dict = feed_for_net)
        if (display):
            print '--overall loss for ', accumulated_step,'layers:', tf_sess.run(self.train_multi_step_loss[accumulated_step - 1], feed_dict = feed_for_net)
            for step in range(accumulated_step):
                print '---loss for prediction #', step, ':', tf_sess.run(\
                                                                         self.mean_square_loss(self.step_train_net_layer[step]["deconv3"],\
                                                                         self.step_train_net_layer[step]["frame_next_img_placeholder"]),\
                                                                         feed_dict = feed_for_net)
