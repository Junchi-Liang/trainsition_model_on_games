from os import listdir
from os.path import isfile, join, isdir
import glob
import skimage.io
import numpy as np
import numpy.random

# This module is for reading data from disk 
# assumption 1: data in input directory is collected according to episodes
# such that each episode has its own directory under input directory
# assumption 2: images in each episode are named by their order, index starts from 0
# assumption 3: rewards and actions are stored as file separately in each episode directory.
# namely, we look act.log for actions and reward.log for rewards in each episode directory

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
        action_file_dict[episode] = join(input_dir, episode + '/act.log')
        reward_file_dict[episode] = join(input_dir, episode + '/reward.log')
    return [episode_list, img_dict, action_file_dict, reward_file_dict]

def get_gray_img_tensor(input_file):
    """
        read an image, and reshape it to 3d tensor with it size as (n, m, 1) when the image is (n, m)
        input_file : str
        input_file - path for input image
    """
    img_input = skimage.io.imread(input_file)
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
             in another word, ds[e][i][m][n][c] is pixel[n, m] in channel c from i-th image of episode e
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
                  dataset[e][i][m][n][c] should be pixel [n, m] in channel c from i-th image of episode e.
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








