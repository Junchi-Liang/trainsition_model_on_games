import numpy as np
import nn_utils.cnn_utils
import scipy.misc
from os import listdir
from os.path import isfile, join, isdir
import glob
import numpy.random

class VOC2011:
    """
        Data Set Pascal VOC 2011
    """
    def __init__(self, dataset_dir, img_height = None, img_width = None, mean_img = None):
        """
            dataset_dir : string
            dataset_dir = directory of dataset. under this directory, 
                          there should be Annotations, ImageSets, JPEGImages, 
                          SegmentationClass and SegmentationObject
            img_height : int
            img_height = when this is not none, image will be resized to this height
            img_width : int
            img_width = when this us not none, image will be resized to this width
            mean_img : np.array
            mean_img = mean value of training set, when this is None, a mean will be computed
        """
        path_list_filename = join(dataset_dir, 'ImageSets/Segmentation')
        train_list_filename = join(path_list_filename, 'train.txt')
        with open (train_list_filename, 'r')  as train_list_file:
            train_lines_raw = train_list_file.read().split('\n')
        train_list_file.close()
        val_list_filename = join(path_list_filename, 'val.txt')
        with open (val_list_filename, 'r') as val_list_file:
            val_lines_raw = val_list_file.read().split('\n')
        val_list_file.close()
        self.dataset_path = dataset_dir
        self.val_index = [line for line in val_lines_raw if len(line) > 0]
        self.train_index = [line for line in train_lines_raw if len(line) > 0]
        self.img_height = img_height
        self.img_width = img_width
        self.segmentation_color = self.color_map()
        self._random_permutation()
        self.index_next = 0
        if (mean_img is None):
            self.mean_img = self.mean_image()
        else:
            self.mean_img = mean_img

    def _random_permutation(self):
        """
            get an internal random permutation
        """
        self.permutation = numpy.random.permutation(range(len(self.train_index)))

    def jpg_image_path(self, index):
        """
            get path to input image from index
        """
        return join(self.dataset_path, 'JPEGImages', index + '.jpg')

    def segmentation_image_path(self, index):
        """
            get path to segmentation image from index
        """
        return join(self.dataset_path, 'SegmentationClass', index + '.png')

    def npz_ground_truth_path(self, index, npz_path = 'NPZGroundTruth'):
        """
            get path to npz ground truth file
        """
        return join(self.dataset_path, npz_path, index + '.npz')

    def color_map(self):
        """
            return a color map for class segmentation
            -----------------------------------------
            return color
            color : list
            color = a list of color, color[i] is color for class i, which is RGB
        """
        color = [[0, 0, 0], 
                 [128, 0, 0], 
                 [0, 128, 0],
                 [128, 128, 0], 
                 [0, 0, 128], 
                 [128, 0, 128], 
                 [0, 128, 128],
                 [128, 128, 128],
                 [64, 0, 0],
                 [192, 0, 0],
                 [64, 128, 0],
                 [192, 128, 0],
                 [64, 0, 128],
                 [192, 0, 128],
                 [64, 128, 128],
                 [192, 128, 128],
                 [0, 64, 0],
                 [128, 64, 0],
                 [0, 192, 0],
                 [128, 192, 0],
                 [0, 64, 128]]
        return color

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
        
    def load_ground_truth(self, img_path, resize = True, interp = 'nearest', color = None):
        """
            load an image from SegmentationClass, convert it to ground truth
            img_path : string
            img_path = path to the segmentation image
            resize : boolean
            resize = indicator for if this image should be resized
            interp : string
            interp = Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic')
            color : list
            color = color map for segmentation image, when this is none, self.segmentation_color is used
            --------------------------------------------------------------------------------------------
            return ground_truth
            ground_truth : numpy.ndarray
            ground_truth = ground truth interpreted from segmentation image, shape (h, w)
        """
        if (color is None):
            seg_col = self.segmentation_color
        else:
            seg_col = color
        raw_image = self.load_image(img_path, resize, interp)
        ground_truth = np.zeros([raw_image.shape[0], raw_image.shape[1]], np.int)
        for i in range(raw_image.shape[0]):
            for j in range(raw_image.shape[1]):
                label = 0
                for k in range(1, len(seg_col)):
                    if (raw_image[i, j, 0] == seg_col[k][0] and raw_image[i, j, 1] == seg_col[k][1] and raw_image[i, j, 2] == seg_col[k][2]):
                        label = k
                        break
                ground_truth[i, j] = label
        return ground_truth

    def has_next(self):
        """
            check if there is next batch
        """
        return (self.index_next < len(self.train_index))

    def next_batch(self, batch_size, load_npz_ground_truth_from = None):
        """
            get next batch, when an epoch finishes, a new random permutation is set and the first batch is returned
            batch_size : int
            batch_size = batch size
            load_npz_ground_truth_from : string
            load_npz_ground_truth_from = when this is not none, load ground truth from this directory
            --------------------------------------------------------------------------------------------
            return [img_set, ground_truth_set]
            img_set : numpy.ndarray
            img_set = image set, shape (batch size, image height, image width, 3)
            ground_truth_set : numpy.ndarray
            ground_truth_set = ground truth set, shape (batch size, image height, image width)
        """
        if (not self.has_next()):
            self._random_permutation()
            self.index_next = 0
        if (self.index_next + batch_size <= len(self.train_index)):
            real_batch_size = batch_size
        else:
            real_batch_size = len(self.train_index) - self.index_next
        img_set = np.zeros([batch_size, self.img_height, self.img_width, 3])
        ground_truth_set = np.zeros([batch_size, self.img_height, self.img_width])
        for i in range(self.index_next, self.index_next + real_batch_size):
            img_index = self.train_index[self.permutation[i]]
            img_input = self.load_image(self.jpg_image_path(img_index))
            if (load_npz_ground_truth_from is None):
                ground_truth = self.load_ground_truth(self.segmentation_image_path(img_index))
            else:
                ground_truth = np.load(self.npz_ground_truth_path(img_index, load_npz_ground_truth_from))["ground_truth"]
            img_set[i - self.index_next] = img_input
            ground_truth_set[i - self.index_next] = ground_truth
        dup_cnt = 0
        while real_batch_size < batch_size:
            if (dup_cnt >= real_batch_size):
                dup_cnt = 0
            img_set[real_batch_size] = img_set[dup_cnt]
            ground_truth_set[real_batch_size] = ground_truth_set[dup_cnt]
            dup_cnt = dup_cnt + 1
            real_batch_size = real_batch_size + 1
        self.index_next = self.index_next + batch_size
        return [img_set, ground_truth_set]

    def val_batch(self, batch_size, load_npz_ground_truth_from = None):
        """
            get a random validation batch
            batch_size : int
            batch_size = batch size
            load_npz_ground_truth_from : string
            load_npz_ground_truth_from = when this is not none, load ground truth from this directory
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
            img_input = self.load_image(self.jpg_image_path(img_index))
            if (load_npz_ground_truth_from is None):
                ground_truth = self.load_ground_truth(self.segmentation_image_path(img_index))
            else:
                ground_truth = np.load(self.npz_ground_truth_path(img_index, load_npz_ground_truth_from))["ground_truth"]
            img_set[i] = img_input
            ground_truth_set[i] = ground_truth
        return [img_set, ground_truth_set]

    def visualize_segmentation(self, seg_output, color = None):
        """
            construct a color image according to segmentation result
            seg_output : np.ndarray
            seg_output = segmentation result, shape (image height, image width)
            ------------------------------------------------------------------------------------------
            return image
            image : np.ndarray
            image = color image as segmentation output, shape (image height, image width, 3)
        """
        if (color is None):
            seg_col = self.segmentation_color
        else:
            seg_col = color
        image = np.zeros([seg_output.shape[0], seg_output.shape[1], 3], np.int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                label = seg_output[i, j]
                image[i, j, 0] = seg_col[label][0]
                image[i, j, 1] = seg_col[label][1]
                image[i, j, 2] = seg_col[label][2]
        return image

    def convert_to_npz_ground_truth(self, npz_path = 'NPZGroundTruth', resize = True):
        """
            convert the segmentation ground truth to npz files
            npz_path : string
            npz_path = directory where npz files are stored
            resize : boolean
            resize = when this is true, images are resized
        """
        for index in (self.train_index + self.val_index):
            filename = self.npz_ground_truth_path(index, npz_path)
            ground_truth = self.load_ground_truth(self.segmentation_image_path(index), resize = resize)
            np.savez(filename, ground_truth = ground_truth)

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
            img_set[i] = self.load_image(self.jpg_image_path(img_index))
        mean_img = np.mean(img_set, axis = 0)
        return mean_img
