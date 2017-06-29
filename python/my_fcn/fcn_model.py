import tensorflow as tf
import numpy as np
import nn_utils.cnn_utils
from my_vgg.vgg16_model import VGG16_model

class FCN_model:
    """
        My Implementation for FCN-8s
    """
    def __init__(self, convert_from_vgg = None, load_from_vgg = None):
        """
            convert_from_vgg : VGG16_model
            convert_from_vgg = the VGG16 model which will be converted to FCN.
                               when an VGG16 model is converted to FCN, they will share the same weight object,
                               i.e. any modification in these weights will appear in both models
            load_from_vgg : VGG16_model
            load_from_vgg = the VGG16 model model which will be loaded to FCN.
                            when an VGG16 model is loaded to FCN, they will keep their own saparate weight objects,
                            i.e. changes of one model will only appear in this model
        """
