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

    def load_image(self, img_path, resize = True, interp = 'nearest'):
        """
            load an image
            img_path : string
            img_path = path to the image
            resize : boolean
            resize = indicator for if this image should be resized.
            interp : string
            interp = Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
            ------------------------------------------------------------------------------------------
            return image
            image : numpy.ndarray
            image = loaded image, shape (h, w, 3), its channels are RGB
        """
        img_raw = scipy.misc.imread(img_path, mode = 'RGB')
        if (resize):
            img = scipy.misc.imresize(img_raw, [self.img_height, self.img_width, 3], interp = interp)
            return img
        else:
            return img_raw

    def load_ground_truth(self, mat_path, resize = True, interp = 'nearest'):
        """
            load a .mat ground truth file
            mat_path : string
            mat_path = path to the .mat file
            resize : boolean
            resize = indicator for if this image should be resized
            interp : string
            interp = Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
            --------------------------------------------------------------------------------------------
            return ground_truth
            ground_truth : numpy.ndarray
            ground_truth = ground truth loaded, shape (h, w)
        """
        raw_mat = scipy.io.loadmat(mat_path)
        raw_ground_truth = raw_mat['GTcls'][0][0][1]
        if (resize):
            ground_truth = scipy.misc.imresize(raw_ground_truth, [self.img_height, self.img_width], interp = interp)
        else:
            ground_truth = raw_ground_truth
        return ground_truth

    def has_next(self):
        """
            check if there is next batch
        """
        return (self.next_train_index < len(self.train_index))

    def next_batch(self, batch_size, mean_img = None):
        """
            get next batch, when an epoch finishes, a new random permutation is set and the first batch is returned
            batch_size : int
            batch_size = batch size
            mean_img : np.array
            mean_img = when this is not none, all image will be substracted by this mean
            --------------------------------------------------------------------------------------------
            return [img_set, ground_truth_set]
            img_set : numpy.ndarray
            img_set = image set, shape (batch size, image height, image width, 3)
            ground_truth_set : numpy.ndarray
            ground_truth_set = ground truth set, shape (batch size, image height, image width)
        """
        if (not self.has_next()):
            self._random_permutation()
            self.next_train_index = 0
