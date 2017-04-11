import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils
from generator_net import Generator_net
from discriminator_net import Discriminator_net
import data_utils.data_read_util

class Gan_net:
    """
        GAN
    """
    def __init__(self, stacked_img_num = 4, image_height = 84, image_width = 84, num_actions = 18, num_channels = 1, training_batch_size_g = 32, test_batch_size_g = None, training_batch_size_d = 64, test_batch_size_d = None):
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
        self.batch_size_train_g = training_batch_size_g
        self.batch_size_test_g = test_batch_size_g
        self.batch_size_train_d = training_batch_size_d
        self.batch_size_test_d = test_batch_size_d

        self.g_net = Generator_net(stacked_img_num, image_height, image_width, num_actions, num_channels, training_batch_size_g, test_batch_size_g)
        self.d_net = Discriminator_net(stacked_img_num + 1, image_height, image_width, num_actions, num_channels, training_batch_size_d, test_batch_size_d)

        # concatenate output and input of generative network as input for discriminator network, which is used for computing loss when we train generative network
        self.concat_layer_g = tf.concat([self.g_net.net_train["deconv3"], self.g_net.net_train["img_stacked_input_placeholder"]], axis = 3)
        self.d_train = self.d_net.construct_network_computational_graph_without_img(self.concat_layer_g, training_batch_size_g)

        lambda_adv_g = 0.5
        lambda_lp = 0.5
        lambda_gdl = 0.5
        self.g_loss_adv = tf.nn.weighted_cross_entropy_with_logits(self.d_train["true_label_placeholder"], self.d_train["fc6"], 1.0)
        self.g_loss_gdl = self.g_net.gradient_difference_loss(self.g_net.net_train["deconv3"], self.g_net.net_train["frame_next_img_placeholder"])
        self.g_loss = lambda_adv_g * self.g_loss_adv + lambda_lp * self.g_net.train_loss + lambda_gdl * self.g_loss_gdl

        self.d_loss = self.d_net.train_loss

        self.train_step_g = tf.train.RMSPropOptimizer(1e-4,
                                                    momentum=0.9).minimize(self.g_loss, var_list=self.g_net.param)
        self.train_step_d = tf.train.RMSPropOptimizer(1e-4,
                                                    momentum=0.9).minimize(self.d_loss, var_list=self.d_net.param)

    def convert_g2d(self, g_net_img_input, g_net_output):
        """
            convert output tensor of generative network to input tensor for discriminative
            g_net_img_input : numpy.array
            g_net_img_input - 4d tensor, input image tensor for generative network.
                              shape (b, m, n, c * s) where b is batch size, (m, n) is image size, c is number of channels, s is number of stacked images
            g_net_output : numpy.array
            g_net_output - 4d tensor, output tensor from generative network.
                           shape (b, m, n, c) where b is batch size, (m, n) is image size, c is number of channels
            return stacked_img_tensor
            stacked_img_tensor : numpy.array
            stacked_img_tensor - 4d tensor, input image tensor for discriminative network. shape (b, m, n, (s + 1) * c)
        """
        stacked_img_tensor = np.concatenate((g_net_output, g_net_img_input), axis = 3)
        return stacked_img_tensor

    def minibatch_g2d(self, batch_g, g_net_output, preprocess = False, mean_img = None):
        """
            get a minibatch for discriminative network from a minibatch of generative network
            batch_g : list
            batch_g - a batch of generative network
            g_net_output : numpy.array
            g_net_output - 4d tensor, output from generative network for batch_g.
                           shape (b, m, n, c)
            preprocess : boolean
            preprocess - if yes, preprocess next frame
            return batch_d = [d_input_imgs, action_tensor, true_labels]
            batch_d : list
            batch_d - a batch for discriminative network.
                      d_input_imgs, 4d tensor, shape (b, m, n, (s + 1) * c)
                      action_tensor, 1d tensor, shape (b, )
                      true_labels, 1d tensor, shape (b, )
        """
        g_input_imgs, reward_tensor, action_tensor, next_img_tensor = batch_g
        stack_0 = self.convert_g2d(g_input_imgs, g_net_output)
        if (preprocess):
            stack_1 = self.convert_g2d(g_input_imgs, self.img_preproprocess(next_img_tensor, mean_img))
        else:
            stack_1 = self.convert_g2d(g_input_imgs, next_img_tensor)
        d_input_imgs = np.concatenate((stack_0, stack_1), axis = 0)
        true_labels = np.concatenate((np.zeros([next_img_tensor.shape[0], 1]), np.zeros([next_img_tensor.shape[0], 1]) + 1), axis = 0)
        batch_d = [d_input_imgs, action_tensor, true_labels]
        return batch_d

    def img_preproprocess(self, img, mean_img):
        """
            preprocess image
            img : numpy.array
            img - image(s) to be preprocessed, it can be one image or a batch of images
            mean_img : numpy.array
            mean_img - mean image of the dataset
            return img_processed
            img_processed : numpy.array
            img_processed - result image
        """
        if (mean_img is None):
            img_processed = img
        else:
            img_processed = (np.float32(img) - np.float32(mean_img)) / 255.0
        return img_processed
        

    def train_iteration(self, tf_sess, num_iteration_g = 1, num_iteration_d = 5, batch_memory_arg = None, batch_disk_arg = None, display = False, mean_img = None, preprocess = False):
        """
        """
        if (batch_memory_arg is not None):
            dataset, episode_list, action_file_dict, reward_file_dict, frame_stride = batch_memory_arg
            for i_d in range(num_iteration_d):
                batch, _ = data_utils.data_read_util.minibatch_from_whole_dataset(dataset, episode_list, action_file_dict, reward_file_dict, self.num_stack, frame_stride, self.batch_size_train_g)
                stacked_img_tensor, reward_tensor, action_tensor, next_img_tensor = batch
                action_one_hot_tensor = data_utils.data_read_util.one_hot_action(action_tensor, self.num_act)
                if (preprocess):
                    processed_input = self.img_preproprocess(stacked_img_tensor, mean_img)
                else:
                    processed_input = stacked_img_tensor
                if (preprocess):
                    processed_output = self.img_preproprocess(next_img_tensor, mean_img)
                else:
                    processed_output = next_img_tensor
                output_d = tf_sess.run(self.g_net.net_train["deconv3"], feed_dict = {self.g_net.net_train["img_stacked_input_placeholder"]: processed_input, self.g_net.net_train["action_input_placeholder"]: action_one_hot_tensor})
                batch_d = self.minibatch_g2d(batch, output_d, preprocess, mean_img)
                d_input_imgs, action_tensor, true_labels = batch_d
                self.train_step_d.run(feed_dict = {self.d_net.net_train["img_stacked_input_placeholder"]: d_input_imgs, self.d_net.net_train["action_input_placeholder"]: action_one_hot_tensor, self.d_net.net_train["true_label_placeholder"]: true_labels})
                if (display):
                    print '--training loss of D, #', i_d, ':'
                # TODO: start here
        elif (batch_disk_arg is not None):
            episode_list, img_dict, action_file_dict, reward_file_dict, frame_stride = batch_disk_arg
        else:
            return








