import cv2
import numpy as np
import scipy.misc

def circle_object(p_x, p_y, c, img_height, img_width):
    """
        p_x : int
        p_x - center of the circle
        p_y : int
        p_y - center of the circle
        c : int
        c - radius of the circle
        img_height : int
        img_height - height of the image
        img_width : int
        img_width - width of the image
        return output
        output : np.ndarray
        output - output image
    """
    output = np.zeros([img_height, img_width], np.uint8)
    output[:, :] = 0
    for i in range(-c, c + 1):
        for j in range(-c, c + 1):
            if (i ** 2 + j ** 2 < c ** 2):
                if (p_x + i >= 0 and p_x + i < img_height and p_y + j >= 0 and p_y + j < img_width):
                    output[i + p_x, j + p_y] = 255
    return output
    
