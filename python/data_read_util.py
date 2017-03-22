from os import listdir
from os.path import isfile, join, isdir
import glob
import skimage.io
import numpy as np

# This module is for reading data from disk 
# assumption 1: data in input directory is collected according to episodes
# such that each episode has its own directory under input directory
# assumption 2: images in each episode are named by their order, index starts from 0
# assumption 3: rewards and actions are stored as file separately in each episode directory.
# namely, we look act.log for actions and reward.log for rewards in each episode directory

def get_file_list(input_dir, img_format):
    """ input_dir - input directory
        img_format - image format, e.g. jpg
        return [episode_list, img_dict, action_file_dict, reward_file_dict]
        episode_list : list
        episode_list - list of name of episode, can be used as indices for img_dict, action_file_dict, reward_file_dict
        img_dict : dictionary
        img_dict - collection of file name of images, complete file name. File names in each episode are sorted.
        action_file_dict : dictionary
        action_file_dict - collection of file name for action files, complete file name
        reward_file_dict : dictionary
        reward_file_dict - collection of file name for reward files, complete file name
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
        img_dict - dictionary of file names of images, complete file name. We assume file names in each epsiode are sorted.
                   so the result will have the same order as this img_dict
        return ds
        ds : dictionary
        ds - the whole data set, indexed by elements in episode_list. Each element in this dictionary is a 4d tensor.
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
    
