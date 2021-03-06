from os import listdir
from os.path import isfile, join, isdir
import glob
import numpy as np
import numpy.random
import scipy.misc

# This module is for reading data from disk 
# assumption 1: data in input directory is collected according to episodes
# such that each episode has its own directory under input directory
# assumption 2: images in each episode are named by their order, index starts from 0
# assumption 3: rewards and actions are stored as file separately in each episode directory.
# namely, we look act_idx.log for actions and reward.log for rewards in each episode directory
# assumption 4: rewards and actions are separated by new line ('\n') in these files

def get_file_list(input_dir, img_format):
    """ 
        input_dir : str
        input_dir - input directory
        img_format : str
        img_format - image format, e.g. jpg
        return [episode_list, img_dict, action_file_dict, reward_file_dict]
        episode_list : list
        episode_list - list of name of episode, can be used as indices for img_dict, action_file_dict, reward_file_dict
        img_dict : dictionary
        img_dict - collection of file name of images, complete file name, indexed by elements in episode_list. File names in each episode are sorted.
        action_file_dict : dictionary
        action_file_dict - collection of file name for action files, complete file name, indexed by elements in episode_list.
        reward_file_dict : dictionary
        reward_file_dict - collection of file name for reward files, complete file name, indexed by elements in episode_list.
    """
    episode_list = [d for d in listdir(input_dir) if isdir(join(input_dir, d))]
    img_dict = {}
    action_file_dict = {}
    reward_file_dict = {}
    for episode in episode_list:
        img_dict[episode] = glob.glob(join(input_dir, episode + "/*." + img_format))
        img_dict[episode].sort(key=lambda file_name: int(file_name.split('/')[-1].split('.')[0]))
        action_file_dict[episode] = join(input_dir, episode + '/act_idx.log')
        reward_file_dict[episode] = join(input_dir, episode + '/reward.log')
    return [episode_list, img_dict, action_file_dict, reward_file_dict]

def get_gray_img_tensor(input_file):
    """
        read an image, and reshape it to 3d tensor with it size as (n, m, 1) when the image is (n, m), n is height, m is width
        input_file : str
        input_file - path for input image
    """
    img_input = scipy.misc.imread(input_file)
    img_tensor = img_input.reshape((img_input.shape[0], img_input.shape[1], 1))
    return img_tensor

def read_whole_gray_dataset(episode_list, img_dict):
    """
        read the whole dataset (assume it only consists of gray scale images) in to memeory
        episode_list : list
        episode_list - list of episodes, which should be indices of img_dict. You can get it from get_file_list
        img_dict : dictionary
        img_dict - dictionary of file names of images, complete file name, indexed by elements in episode_list.
                   we assume file names in each episode are sorted. so the result will have the same order as this img_dict.
                   you can get it from get_file_list
        return ds
        ds : dictionary
        ds - the whole image set, indexed by elements in episode_list. Each element in this dictionary is a 4d tensor.
             in another word, ds[e][i, m, n, c] is pixel[n, m] in channel c from i-th image of episode e
    """
    ds = {}
    cnt = 0
    whole_size = len(episode_list)
    for episode in episode_list:
        cnt = cnt + 1
        print 'reading episode ', episode, ' (', cnt, ' / ', whole_size, ')'
        ds[episode] = np.stack((x for x in [get_gray_img_tensor(f) for f in img_dict[episode]]))
    return ds

def minibatch_from_whole_dataset(dataset, episode_list, action_file_dict, reward_file_dict, stacked_img_num, frame_stride, batch_size):
    """
        get a random minibatch from dataset in memory
        dataset : dictionary
        dataset - the whole image set. Generally, you would like to get it from read_whole_gray_dataset. 
                  It is assumed that this dictionary is indexed by elements in episode_list, 
                  while each element in this dictionary is a 4d tensor.
                  dataset[e][i, m, n, c] should be pixel [n, m] in channel c from i-th image of episode e.
                  you can get it from read_whole_gray_dataset
        episode_list : list
        episode_list - list of episodes, which should be indices of img_dict. You can get it from get_file_list
        action_file_dict : dictionary
        action_file_dict - collection of file name for action files, complete file name, indexed by elements in episode_list.
                           you can get it from get_file_list
        reward_file_dict : dictionary
        reward_file_dict - collection of file name for action files, complete file name, indexed by elements in episode_list.
                           you can get it from get_file_list
        stacked_img_num : int
        stacked_img_num - number of stacked image for one frame. it should be larger than one, as it includes starting frame.
        frame_stride : int
        frame_stride - stride for sampling frames. if you want to stack every frames, set it to 1.
                       if you want to get 1 frame of every 2 frames, set it to 2.
        batch_size : int
        batch_size - size of the batch
        return [batch, selected_index]
        selected_index : list
        selected_index - selected frame indices. each element in this list is a list with size of 2.
                         selected_index[i][0] is episode of the i-th selected images while selected_index[i][1] is the index of starting frame,
                         so we sample from selected_index[i][1], and then selected_index[i][1] - frame_stride, selected_index[i][1] - 2 * frame_stride, ...
        batch : list
        batch = [stacked_img_tensor, reward_tensor, action_tensor, next_img_tensor]
        stacked_img_tensor : numpy.ndarray
        stacked_img_tensor - stacked input image, with the same order as selected_index. 
                             4d tensor, its shape is [batch_size][m][n][c * stacked_img_num] where [m, n] is the size of the image,
                             the number of channels of images is c.
                             i.e. stacked_img_tensor[0, x, y, 0] is the pixel[x, y] from the first channel of the first image set (each set is stacked images),
                             stacked_img_tensor[1, a, b, stacked_img_num * c - 1] is pixel [a, b] from the last channel of the oldest image in second image set.
       reward_tensor : tuple
       reward_tensor - 1d list, shape (batch_size,). the collection of rewards, in the same order as selected_index
       action_tensor : tuple
       action_tensor - 1d list, shape (batch_size,). the collection of actions, in the same order as selected_index
       next_img_tensor : numpy.ndarray
       next_img_tensor - collection of next frames.
                         4d tensor, it shape is (batch_size, m, n, c) where [m, n] is the size of the image, c is the number of channels of images.
    """
    selected_index = []
    for _ in range(0, batch_size):
        episode = numpy.random.choice(episode_list)
        episode_size = dataset[episode].shape[0]
        selected_index.append([episode, numpy.random.choice(range(frame_stride * (stacked_img_num - 1), episode_size - 1))])
    selected_index.sort(key=lambda t: t[0])
    stacked_img_list = []
    next_img_list = []
    previous_episode = ''
    rewards_cur_episode = []
    actions_cur_episode = []
    reward_column = []
    action_column = []
    for index_now in selected_index:
        episode, frame = index_now
        if (episode != previous_episode):
            reward_file = open(reward_file_dict[episode], 'r')
            rewards_raw = reward_file.read().split('\n')
            rewards_cur_episode = [float(r) for r in rewards_raw if len(r) > 0]
            action_file = open(action_file_dict[episode], 'r')
            actions_raw = action_file.read().split('\n')
            actions_cur_episode = [float(a) for a in actions_raw if len(a) > 0]
            previous_episode = episode
        reward_column.append(rewards_cur_episode[frame])
        action_column.append(actions_cur_episode[frame])
        next_img_list.append(dataset[episode][frame + 1])
        previous_img_list = [dataset[episode][frame - frame_cnt * frame_stride] for frame_cnt in range(0, stacked_img_num)]
        img_size_0 = dataset[episode].shape[1]
        img_size_1 = dataset[episode].shape[2]
        channel_num = stacked_img_num * dataset[episode].shape[3]
        stacked_img_list.append(np.stack((img for img in previous_img_list), axis=2).reshape(img_size_0, img_size_1, channel_num))
    stacked_img_tensor = np.stack((stacked_img for stacked_img in stacked_img_list))
    reward_tensor = np.stack((r for r in reward_column))
    action_tensor = np.stack((a for a in action_column))
    next_img_tensor = np.stack((next_img for next_img in next_img_list))
    batch = [stacked_img_tensor, reward_tensor, action_tensor, next_img_tensor]
    return [batch, selected_index]

def minibatch_from_disk(episode_list, img_dict, action_file_dict, reward_file_dict, stacked_img_num, frame_stride, batch_size):
    """
        get a minibatch from disk
        episode_list : list
        episode_list - list of episodes, which should be indices of img_dict. You can get it from get_file_list
        img_dict : dictionary
        img_dict - collection of file name of images, complete file name, indexed by elements in episode_list.
                   File names in each episode are sorted. you can get it from get_file_list
        action_file_dict : dictionary
        action_file_dict - collection of file name for action files, complete file name, indexed by elements in episode_list.
                           you can get it from get_file_list
        reward_file_dict : dictionary
        reward_file_dict - collection of file name for action files, complete file name, indexed by elements in episode_list.
                           you can get it from get_file_list
        stacked_img_num : int
        stacked_img_num - number of stacked image for one frame. it should be larger than one, as it includes starting frame.
        frame_stride : int
        frame_stride - stride for sampling frames. if you want to stack every frames, set it to 1.
                       if you want to get 1 frame of every 2 frames, set it to 2.
        batch_size : int
        batch_size - size of the batch
        return [batch, selected_index]
        selected_index : list
        selected_index - selected frame indices. each element in this list is a list with size of 2.
                         selected_index[i][0] is episode of the i-th selected images while selected_index[i][1] is the index of starting frame,
                         so we sample from selected_index[i][1], and then selected_index[i][1] - frame_stride, selected_index[i][1] - 2 * frame_stride, ...
        batch : list
        batch = [stacked_img_tensor, reward_tensor, action_tensor, next_img_tensor]
        stacked_img_tensor : numpy.ndarray
        stacked_img_tensor - stacked input image, with the same order as selected_index. 
                             4d tensor, its shape is [batch_size][m][n][c * stacked_img_num] where [m, n] is the size of the image,
                             the number of channels of images is c.
                             i.e. stacked_img_tensor[0, x, y, 0] is the pixel[x, y] from the first channel of the first image set (each set is stacked images),
                             stacked_img_tensor[1, a, b, stacked_img_num * c - 1] is pixel [a, b] from the last channel of the oldest image in second image set.
       reward_tensor : tuple
       reward_tensor - 1d list, shape (batch_size,). the collection of rewards, in the same order as selected_index
       action_tensor : tuple
       action_tensor - 1d list, shape (batch_size,). the collection of actions, in the same order as selected_index
       next_img_tensor : numpy.ndarray
       next_img_tensor - collection of next frames.
                         4d tensor, it shape is (batch_size, m, n, c) where [m, n] is the size of the image, c is the number of channels of images.
    """
    selected_index = []
    for _ in range(0, batch_size):
        episode = numpy.random.choice(episode_list)
        episode_size = len(img_dict[episode])
        selected_index.append([episode, numpy.random.choice(range(frame_stride * (stacked_img_num - 1), episode_size - 1))])
    selected_index.sort(key=lambda t: t[0]) 
    previous_episode = ''
    stacked_img_list = []
    next_img_list = []
    rewards_cur_episode = []
    actions_cur_episode = []
    reward_column = []
    action_column = []
    for index_now in selected_index:
        episode, frame = index_now
        if (episode != previous_episode):
            reward_file = open(reward_file_dict[episode], 'r')
            rewards_raw = reward_file.read().split('\n')
            rewards_cur_episode = [float(r) for r in rewards_raw if len(r) > 0]
            action_file = open(action_file_dict[episode], 'r')
            actions_raw = action_file.read().split()
            actions_cur_episode = [float(a) for a in actions_raw if len(a) > 0]
            previous_episode = episode
        reward_column.append(rewards_cur_episode[frame])
        action_column.append(actions_cur_episode[frame])
        next_img_list.append(get_gray_img_tensor(img_dict[episode][frame + 1]))
        previous_img_list = [get_gray_img_tensor(img_dict[episode][frame - frame_cnt * frame_stride]) for frame_cnt in range(0, stacked_img_num)]
        img_size_0 = previous_img_list[0].shape[0]
        img_size_1 = previous_img_list[0].shape[1]
        channel_num = stacked_img_num * previous_img_list[0].shape[2]
        stacked_img_list.append(np.stack((img for img in previous_img_list), axis = 2).reshape(img_size_0, img_size_1, channel_num))
    stacked_img_tensor = np.stack((stacked_img for stacked_img in stacked_img_list))
    reward_tensor = np.stack((r for r in reward_column))
    action_tensor = np.stack((a for a in action_column))
    next_img_tensor = np.stack((next_img for next_img in next_img_list))
    batch = [stacked_img_tensor, reward_tensor, action_tensor, next_img_tensor]
    return [batch, selected_index]

def one_hot_action(action_tensor, num_actions):
    """
        convert a 1d tensor to a collections of one hot vector for actions
        action_tensor : 1d tensor or list
        action_tensor - list of actions, which can be obtained from a minibatch. We assume all actions locate in [0, num_actions - 1].
        num_actions : int
        num_actions - number of actions
    """
    actions_list = []
    for action in action_tensor:
        actions_list.append([(1 if int(action) == a else 0) for a in range(0, num_actions)])
    return np.array(actions_list)

def mean_img_from_disk(sample_size, episode_list, img_dict):
    """
        get mean images from sampling on images in disk
        sample_size : int
        sample_size - number of sampling
        episode_list : list
        episode_list - list of episodes, which should be indices of img_dict. You can get it from get_file_list
        img_dict : dictionary
        img_dict - collection of file name of images, complete file name, indexed by elements in episode_list.
                   File names in each episode are sorted. you can get it from get_file_list
        return mean_img
        mean_img : numpy.array
        mean_img - 3d tensor, shape (m, n, c)
    """
    chosen_img_list = []
    for _ in range(sample_size):
        episode_selected = numpy.random.choice(episode_list)
        chosen_img_list.append(get_gray_img_tensor(img_dict[episode_selected][numpy.random.choice(range(len(img_dict[episode_selected])))]))
    chosen_imgs = np.stack((img for img in chosen_img_list))
    return np.mean(np.float32(chosen_imgs), axis=0)

def minibatch_multi_step_selection(episode_list, stacked_img_num, frame_stride, batch_size, prediction_step, img_dict = None, dataset = None):
    """
    """
    selected_index = []
    for _ in range(0, batch_size):
        episode = numpy.random.choice(episode_list)
        if (img_dict is not None):
            episode_size = len(img_dict[episode])
        else:
            episode_size = dataset[episode].shape[0]
        selected_index.append([episode, numpy.random.choice(range(frame_stride * (stacked_img_num - 1), episode_size - prediction_step))])
    selected_index.sort(key=lambda t: t[0])
    return selected_index

def minibatch_multi_step(episode_list, action_file_dict, reward_file_dict, stacked_img_num, frame_stride, batch_size,  prediction_step, img_dict = None, dataset = None, selected_index = None):
    """
    """
    if (selected_index is None):
        random_selection = minibatch_multi_step_selection(episode_list, stacked_img_num, frame_stride, batch_size, prediction_step, img_dict = img_dict, dataset = dataset)
    else:
        random_selection = selected_index
    previous_episode = ''
    stacked_img_list = []
    ground_truth_list = []
    rewards_cur_episode = []
    actions_cur_episode = []
    reward_column = []
    action_column = []
    for index_now in random_selection:
        episode, frame = index_now
        if (episode != previous_episode):
            reward_file = open(reward_file_dict[episode], 'r')
            rewards_raw = reward_file.read().split('\n')
            rewards_cur_episode = [float(r) for r in rewards_raw if len(r) > 0]
            action_file = open(action_file_dict[episode], 'r')
            actions_raw = action_file.read().split()
            actions_cur_episode = [float(a) for a in actions_raw if len(a) > 0]
            previous_episode = episode
        reward_column.append(rewards_cur_episode[frame : frame + prediction_step])
        action_column.append(actions_cur_episode[frame : frame + prediction_step])
        if (dataset is None):
            multi_step_ground_truth_list = [get_gray_img_tensor(img_dict[episode][frame + frame_predicted]) for frame_predicted in range(1, prediction_step + 1)]
            previous_img_list = [get_gray_img_tensor(img_dict[episode][frame - frame_cnt * frame_stride]) for frame_cnt in range(0, stacked_img_num)]
        else:
            multi_step_ground_truth_list = [dataset[episode][frame + frame_predicted] for frame_predicted in range(1, prediction_step + 1)]
            previous_img_list = [dataset[episode][frame - frame_cnt * frame_stride] for frame_cnt in range(0, stacked_img_num)]
        img_size_0 = previous_img_list[0].shape[0]
        img_size_1 = previous_img_list[0].shape[1]
        channel_num = stacked_img_num * previous_img_list[0].shape[2]
        stacked_img_list.append(np.stack((img for img in previous_img_list), axis = 2).reshape(img_size_0, img_size_1, channel_num))
        channel_num = prediction_step * previous_img_list[0].shape[2]
        ground_truth_list.append(np.stack((img for img in multi_step_ground_truth_list), axis = 2).reshape(img_size_0, img_size_1, channel_num))
    stacked_img_tensor = np.stack((stacked_img for stacked_img in stacked_img_list))
    ground_truth_tensor = np.stack((multi_step_prediction for multi_step_prediction in ground_truth_list))
    reward_tensor = np.array(reward_column, ndmin = 2)
    action_tensor = np.array(action_column, ndmin = 2)
    batch = [stacked_img_tensor, reward_tensor, action_tensor, ground_truth_tensor]
    return [batch, random_selection]

