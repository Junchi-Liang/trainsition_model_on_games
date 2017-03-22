from os import listdir
from os.path import isfile, join, isdir
import glob

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
        img_dict - collection of file name of images, complete file name
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
