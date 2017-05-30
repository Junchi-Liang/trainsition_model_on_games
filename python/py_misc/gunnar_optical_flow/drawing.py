import cv2
import numpy as np
import scipy.misc

def circle_object(p_x, p_y, c, img_height, img_width):
    """
        generate a picture with a circle
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

def random_motion_pictures(speed_max, c, img_height, img_width):
    """
        generate two consecutive image of an moving object
        img_height : int
        img_height - height of the image
        img_width : int
        img_width - width of the image
        speed_max : int
        speed_max - |x1 - x1'| < speed_max and |x2 - x2'| < speed_max
        c : int
        c : radius of the object
        return [img_prev, img_next]
        img_prev : np.ndarray
        img_prev - first image
        img_next : np.ndarray
        img_next - next image
    """
    x1 = np.random.choice(img_height, 1)[0]
    x2 = np.random.choice(img_width, 1)[0]
    while (True):
        x1_next = x1 + np.random.choice(speed_max * 2 + 1, 1)[0] - speed_max
        x2_next = x2 + np.random.choice(speed_max * 2 + 1, 1)[0] - speed_max
        if (x1_next >= 0 and x1_next < img_height and x2_next >= 0 and x2_next < img_width):
            img_next = circle_object(x1_next, x2_next, c, img_height, img_width)
            break
    img_prev = circle_object(x1, x2, c, img_height, img_width)
    return [img_prev, img_next]
