mport numpy as np
import nn_utils.cnn_utils
import scipy.misc
import scipy.io
from os import listdir
from os.path import isfile, join, isdir
import glob
import numpy.random

class SBD:
    """
        Dataset Reader: Semantic Boundaries Dataset
        http://home.bharathh.info/pubs/codes/SBD/download.html
        g = x['GTcls'][0][0][1]
    """
    def __init__(self, dataset_dir, img_height = None, img_width = None, mean_img = None):
        """
            dataset_dir : string
            dataset_dir = directory of dataset. under this directory, 
                          there should be cls, img, inst, train.txt and val.txt.
            img_height : int
            img_height = when this is not none, image will be resized to this height
            img_width : int
            img_width = when this us not none, image will be resized to this width
            mean_img : np.array
            mean_img = mean value of training set, when this is None, a mean will be computed
        """
        self.dataset_path = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        train_list_filename = join(dataset_dir, 'train.txt')
        with open(train_list_filename) as train_list_file:
            train_lines_raw = train_list_file.read().split('\n')
        train_list_file.close()
        self.train_index = [line for line in train_lines_raw if len(line) > 0]
        val_list_filename = join(dataset_dir, 'val.txt')
        with open(val_list_filename) as val_list_file:
            val_lines_raw = val_list_file.read().split('\n')
        val_list_file.close()
        self.val_index = [line for line in val_lines_raw if len(line) > 0]
        self.next_train_index = 0
        self._random_permutation()

    def _random_permutation(self):
        """
            get an internal random permutation
        """
        self.permutation = numpy.random.permutation(range(len(self.train_index)))

    def raw_image_path(self, index):
        """
            get path to input image from index
        """
        return join(self.dataset_path, 'img', index + '.jpg')

    def ground_truth_path(self, index):
        """
            get path to ground truth .mat files
        """
       return join(self.dataset_path, 'cls', index + '.mat')
