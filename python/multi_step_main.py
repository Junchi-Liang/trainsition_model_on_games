import tensorflow as tf
import data_utils.data_read_util
import nn_utils.cnn_utils
import numpy as np
from gan_enc_dec.gan_net import Gan_net
from gan_enc_dec.multi_step_net import Multi_Step_net 

episode_list, img_dict, action_file_dict, reward_file_dict = data_utils.data_read_util.get_file_list('data/raw_data/train', 'jpg')

stacked_img_num = 4
frame_stride = 1
batch_size = 32

prediction_step = 4

net_mul = Multi_Step_net(step_prediction = prediction_step)

ds = data_utils.data_read_util.read_whole_gray_dataset(episode_list, img_dict)
mean_img = data_utils.data_read_util.mean_img_from_disk(500, episode_list, img_dict)

arg_for_batch = [episode_list, action_file_dict, reward_file_dict, stacked_img_num, frame_stride, batch_size,  prediction_step, img_dict, ds, None]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

num_iter = 1000
accumulated_step = 1

for i in range(num_iter):
    if (i % 10 == 0 or i == num_iter - 1):
        print '-iteration #', i
        net_mul.train_iteration(tf_sess = sess, accumulated_step = accumulated_step, arg_for_batch = arg_for_batch, arg_for_normalization = mean_img, display = True)
    else:
        net_mul.train_iteration(tf_sess = sess, accumulated_step = accumulated_step, arg_for_batch = arg_for_batch, arg_for_normalization = mean_img, display = False)




