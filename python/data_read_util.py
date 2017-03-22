from os import listdir
from os.path import isfile, join, isdir

checked_dir = 'data/raw_data/train/'
episode_list = [d for d in listdir(checked_dir) if isdir(join(checked_dir, d))]
print episode_list
