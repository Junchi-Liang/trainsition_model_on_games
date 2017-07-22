import numpy as np
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
    def __init__(self, dataset_dir, img_height = None, img_width = None, mean_img = None, load_to_memory = True):
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
            load_to_memory : boolean
            load_to_memory = when this is True, all images and ground truth will be loaded to memory
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
        self.in_memory = load_to_memory
        if (load_to_memory):
            self.img_all, self.ground_truth_all = self.load_all_to_memory(self.train_index + self.val_index)

    def load_all_to_memory(self, index_list, display = True):
        """
            load all images and ground truth to memory
            index_list : list
            index_list = list of index
            display : boolean
            display = when this is True, display proceeding
            ----------------------------------------------
            return [img_all, ground_truth_all]
            img_all : dictionay
            img_all = collection of raw images, index is the same as items in index_list
            ground_truth_all : dictionay
            ground_truth_all = collection of ground truth, index is the same as items in index_list
        """
        img_all = {}
        ground_truth_all = {}
        for i in range(len(index_list)):
            index = index_list[i]
            img_all[index] = self.load_image(self.raw_image_path(index), resize = False)
            ground_truth_all[index] = self.load_ground_truth(self.ground_truth_path(index), resize = False)
            if (display):
                print 'loading', i + 1, '/', len(index_list)
        return [img_all, ground_truth_all]

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

    def load_image(self, img_path = None, load_index = None, resize = True, interp = 'nearest'):
        """
            load an image
            img_path : string
            img_path = path to the image
            load_index : string
            load_index = when this is not None, load image with this index from memory
            resize : boolean
            resize = indicator for if this image should be resized.
            interp : string
            interp = Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
            ------------------------------------------------------------------------------------------
            return image
            image : numpy.ndarray
            image = loaded image, shape (h, w, 3), its channels are RGB
        """
        if (load_index is None):
            img_raw = scipy.misc.imread(img_path, mode = 'RGB')
        else:
            img_raw = self.img_all[load_index]
        if (resize):
            img = scipy.misc.imresize(img_raw, [self.img_height, self.img_width, 3], interp = interp)
            return img
        else:
            return img_raw

    def load_ground_truth(self, mat_path = None, load_index = None, resize = True, interp = 'nearest'):
        """
            load a .mat ground truth file
            mat_path : string
            mat_path = path to the .mat file
            load_index : string
            load_index = when this is not None, load ground truth with this index from memory
            resize : boolean
            resize = indicator for if this image should be resized
            interp : string
            interp = Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
            --------------------------------------------------------------------------------------------
            return ground_truth
            ground_truth : numpy.ndarray
            ground_truth = ground truth loaded, shape (h, w)
        """
        if (load_index is None):
            raw_mat = scipy.io.loadmat(mat_path)
            raw_ground_truth = raw_mat['GTcls'][0][0][1]
        else:
            raw_ground_truth = self.ground_truth_all[load_index]
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
        if (self.next_train_index + batch_size < len(self.train_index)):
            real_batch_size = batch_size
        else:
            real_batch_size = len(self.train_index) - self.next_train_index
        img_set = np.zeros([batch_size, self.img_height, self.img_width, 3])
        ground_truth_set = np.zeros([batch_size, self.img_height, self.img_width])
        for i in range(self.next_train_index, self.next_train_index + real_batch_size):
            img_index = self.train_index[self.permutation[i]]
            if (self.in_memory):
                img_input = self.load_image(None, img_index)
                ground_truth = self.load_ground_truth(None, img_index)
            else:
                img_input = self.load_image(self.raw_image_path(img_index))
                ground_truth = self.load_ground_truth(self.ground_truth_path(img_index))
            if (mean_img is None):
                img_set[i - self.next_train_index] = img_input
            else:
                img_set[i - self.next_train_index] = img_input - mean_img
            ground_truth_set[i - self.next_train_index] = ground_truth
        dup_cnt = 0
        while (real_batch_size < batch_size):
            img_set[real_batch_size] = img_set[dup_cnt]
            ground_truth_set[real_batch_size] = ground_truth_set[dup_cnt]
            real_batch_size = real_batch_size + 1
            dup_cnt = dup_cnt + 1
        self.next_train_index = self.next_train_index + batch_size
        return [img_set, ground_truth_set]

    def val_batch(self, batch_size, mean_img = None):
        """
            get a random validation batch
            batch_size : int
            batch_size = batch size
            mean_img : np.array
            mean_img = when this is not None, all image will be substracted by this mean
            -------------------------------------------------------------------------------------------
            return [img_set, ground_truth_set]
            img_set : numpy.ndarray
            img_set = image set, shape (batch size, image height, image width, 3)
            ground_truth_set : numpy.ndarray
            ground_truth_set = ground truth set, shape (batch size, image height, image width)
        """
        chosen = numpy.random.permutation(range(len(self.val_index)))[0 : batch_size]
        img_set = np.zeros([batch_size, self.img_height, self.img_width, 3])
        ground_truth_set = np.zeros([batch_size, self.img_height, self.img_width])
        for i in range(batch_size):
            img_index = self.val_index[chosen[i]]
            if (self.in_memory):
                img_input = self.load_image(None, img_index)
                ground_truth = self.load_ground_truth(None, img_index)
            else:
                img_input = self.load_image(self.raw_image_path(img_index))
                ground_truth = self.load_ground_truth(self.ground_truth_path(img_index))
            if (mean_img is None):
                img_set[i] = img_input
            else:
                img_set[i] = img_input - mean_img
            ground_truth_set[i] = ground_truth
        return [img_set, ground_truth_set]

    def mean_image(self, index_list = None):
        """
            get the mean value
            index_list : list
            index_list = when this not none, the mean of images from this list will be returned.
                         otherwise, self.train_index will be used
            -------------------------------------------------------------------------------------
            return mean_img
            mean_img : np.array
            mean_img = mean value of images from the list
        """
        if (index_list is None):
            index_list = self.train_index
        img_set = np.zeros([len(index_list), self.img_height, self.img_width, 3], np.float)
        for i in range(len(index_list)):
            img_index = index_list[i]
            img_set[i] = self.load_image(self.raw_image_path(img_index))
        mean_img = np.mean(img_set, axis = 0)
        return mean_img

