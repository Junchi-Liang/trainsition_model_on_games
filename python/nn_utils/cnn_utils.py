import tensorflow as tf

def normal_weight_variable(shape, stddev_input):
    """
        get a weight for convolution layer from normal distribution where the mean is 0
        shape : list
        shape - the shape of the weight, a len of size 4.
                shape[0], shape[1] are patch size
                shape[2] is the number of input channels
                shape[3] is the number of output channels
        stddev_input : float
        stddev_input - standard deviation for the normal distribution
    """
    initial = tf.truncated_normal(shape, stddev=stddev_input)
    return tf.Variable(initial)

def bias_variable(shape, constant_init):
    """
        get a bias variable for convolution layer
        shape : list
        shape - shape of the bias.
                generally, it should be [c] where c is the number of output channels for the convolution layer.
        constant_init : float
        constant_init - constant for initializing this bias term
    """
    initial = tf.constant(constant_init, shape=shape)
    return tf.Variable(initial, trainable=True)

def weight_convolution_normal(patch_size, num_input_channels, num_output_channels, stddev_input):
    """
        wrap normal_weight_variable
        patch_size : list
        patch_size - patch size is [patch_size[0], patch_size[1]], patch_size[0] = filter height, patch_size[1] = filter width
    """
    return normal_weight_variable([patch_size[0], patch_size[1], num_input_channels, num_output_channels], stddev_input)

def weight_deconvolution_normal(patch_size, num_input_channels, num_output_channels, stddev_input):
    """
        wrap normal_weight_variable
        patch_size : list
        patch_size - patch size is [patch_size[0], patch_size[1]], patch_size[0] = filter height, patch_size[1] = filter width
    """
    return normal_weight_variable([patch_size[0], patch_size[1], num_output_channels, num_input_channels], stddev_input)

def bias_convolution(num_output_channels, constant_init):
    """
        wrap bias_variable
        num_output_channels : int
        num_output_channels - number of output channels
    """
    return bias_variable([num_output_channels], constant_init)
