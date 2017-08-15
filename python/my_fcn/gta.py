import numpy as np
import nn_utils.cnn_utils
import scipy.misc
import scipy.io
from os import listdir
from os.path import isfile, join, isdir
import glob
import numpy.random

class GTA_DS:
    """
        Dataset Reader for GTA data
        https://download.visinf.tu-darmstadt.de/data/from_games/
    """
    def __init__(self, dataset_dir, img_height = 224, img_width = 224):
        """
            dataset_dir : string
            dataset_dir = directory of the GTA dataset.
                          data, scripts are expected under this path.
            img_height : int
            img_height = height of image, when a batch of data is generated, 
                         they will be resized to this height.
            img_width : int
            img_width = width of image, when a batch of data is generated,
                        they will be resized to this width.
        """
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.train_list, self.test_list, self.val_list = self.split_list()
        self._random_permutation()
        self.train_next = 0
        self._standard_mapping()

    def _standard_mapping(self):
        """
            get color mapping for cityscapesMap and camvidMap
        """
        mapping_raw = scipy.io.loadmat(join(dataset_dir), 'scripts/mapping.mat')
        self.camvidMap = mapping_raw['camvidMap'] * 255
        self.cityscapesMap = mapping_raw['cityscapesMap'] * 255

    def _random_permutation(self):
        """
            get an internal random permutation
        """
        self.train_permutation = numpy.random.permutation(range(len(self.train_list)))

    def split_list(self, num_bit = 5):
        """
            get training list, test list, and validation list
            num_bit : int
            num_bit = length of file name
            -----------------------------------------------------
            return [training_list, test_list, val_list]
            training_list
        """
        jpg_list = listdir(join(self.dataset_dir, 'data/jpg_images'))
        label_list = listdir(join(self.dataset_dir, 'data/label_mat'))
        split_raw = scipy.io.loadmat(join(self.dataset_dir, 'scripts/split.mat'))
        train_raw = split_raw['trainIds']
        test_raw = split_raw['testIds']
        val_raw = split_raw['valIds']
        pattern_index = "%0" + str(num_bit) + "d"
        training_list = []
        for i in range(train_raw.shape[0]):
            train_id = pattern_index % train_raw[i, 0]
            if (((train_id + '.jpg') in jpg_list) and ((train_id + '.mat') in label_list)):
                training_list.append(train_id)
        test_list = []
        for i in range(test_raw.shape[0]):
            test_id = pattern_index % test_raw[i, 0]
            if (((test_id + '.jpg') in jpg_list) and ((test_id + '.mat') in label_list)):
                test_list.append(test_id)
        val_list = []
        for i in range(val_raw.shape[0]):
            val_id = pattern_index % val_raw[i, 0]
            if (((val_id + '.jpg') in jpg_list) and ((val_id + '.mat') in label_list)):
                val_list.append(val_id)
        return [training_list, test_list, val_list]

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

    def load_ground_truth(self, mat_path, resize = True, interp = 'nearest')
        """
            load a ground truth mat file
            mat_path : string
            mat_path = path to the segmentation image
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
        g_raw = scipy.io.loadmat(mat_path)['label']
        if (resize):
            return scipy.misc.imresize(g_raw, [self.img_height, self.img_width], interp = interp)
        else:
            return g_raw

    def has_next_train(self):
        """
            check if there is a next training batch
        """
        return (self.train_next < len(self.train_list))

    def next_train_batch(self, batch_size):
        """
            get next batch, when an epoch finishes, a new random permutation is set and the first batch is returned
            batch_size : int
            batch_size = batch size
            load_npz_ground_truth_from : string
            load_npz_ground_truth_from = when this is not none, load ground truth from this directory
            mean_img : np.array
            mean_img = when this is not None, all image will be substracted by this mean
            --------------------------------------------------------------------------------------------
            return [img_set, ground_truth_set]
            img_set : numpy.ndarray
            img_set = image set, shape (batch size, image height, image width, 3)
            ground_truth_set : numpy.ndarray
            ground_truth_set = ground truth set, shape (batch size, image height, image width)
        """
        if (not self.has_next_train()):
            self._random_permutation()
            self.train_next = 0
        if (self.train_next + batch_size <= len(self.train_list)):
            real_batch_size = batch_size
        else:
            real_batch_size = len(self.train_list) - self.train_next
        img_set = np.zeros([batch_size, self.img_height, self.img_width, 3])
        ground_truth_set = np.zeros([batch_size, self.img_height, self.img_width])
        for i in range(self.train_next, self.train_next + real_batch_size):
            train_ind = self.train_list[self.train_permutation[i]]
            img_path = join(self.dataset_dir, 'data/jpg_images', train_ind + '.jpg')
            img_set[i - self.train_next] = self.load_image(img_path)
            mat_path = join(self.dataset_dir, 'data/label_mat', train_ind + '.mat')
            ground_truth_set[i - self.train_next] = self.load_ground_truth(mat_path)
        dup_cnt = 0
        while (real_batch_size < batch_size):
            img_set[real_batch_size] = img_set[dup_cnt]
            ground_truth_set[real_batch_size] = ground_truth_set[dup_cnt]
            dup_cnt = dup_cnt + 1
            real_batch_size = real_batch_size + 1
        self.train_next = self.train_next + batch_size
        return [img_set, ground_truth_set]

    def val_batch(self, batch_size):
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
        chosen = numpy.random.permutation(range(len(self.val_list)))[0 : batch_size]
        img_set = np.zeros([batch_size, self.img_height, self.img_width, 3])
        ground_truth_set = np.zeros([batch_size, self.img_height, self.img_width])
        for i in range(batch_size):
            val_ind = self.val_list[chosen[i]]
            img_path = join(self.dataset_dir, 'data/jpg_images', val_ind + '.jpg')
            img_set[i] = self.load_image(img_path)
            mat_path = join(self.dataset_dir, 'data/label_mat', val_ind + '.mat')
            ground_truth_set[i] = self.load_ground_truth(mat_path)
        return [img_set, ground_truth_set]
