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

def add_circle_to_image(I, c1, c2, r, c):
    """
        add a circle to an image
        I : np.array
        I - 2d array, the image
        c1 : int
        c1 - center of the circle
        c2 : int
        c2 - center of the circle
        r : int
        r - radius of the circle
        c : int
        c - grayscale value of the circle
    """
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            if (i ** 2 + j ** 2 < c ** 2):
                x1 = c1 + i
                x2 = c2 + j
                if (x1 >= 0 and x1 < I.shape[0] and x2 >= 0 and x2 < I.shape[1]):
                    I[x1, x2] = c

def add_triangle_to_image(I, c1, c2, h, c):
    """
        add a triangle to an image
        I : np.array
        I - 2d array, the image
        c1 : int
        c1 : location of the triangle
        c2 : int
        c2 : location of the triangle
        h : int
        h - height of the triangle
        c : int
        c - grayscale value of the triangle
    """
    for i in range(0, h):
        for j in range(0, h - i):
            x2 = c2 - i
            x1 = c1 + j
            if (x1 >= 0 and x1 < I.shape[0] and x2 >= 0 and x2 < I.shape[1]):
                I[x1, x2] = c
            x2 = c2 + i
            if (x1 >= 0 and x1 < I.shape[0] and x2 >= 0 and x2 < I.shape[1]):
                I[x1, x2] = c

def add_square_to_image(I, c1, c2, r, c):
    """
        add a square to the image
        I : np.array
        I - 2d array, the image
        c1 : int
        c1 : location of the square
        c2 : int
        c2 : location of the square
        h : int
        h - size of the square
        c : int
        c - grayscale value of the square
    """
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            if (abs(i) + abs(j) <= r):
                x1 = c1 + i
                x2 = c2 + j
                if (x1 >= 0 and x1 < I.shape[0] and x2 >= 0 and x2 < I.shape[1]):
                    I[x1, x2] = c

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

def random_multiple_moving_objects(num_obj, speed_max, size_max, img_height, img_width):
    """
       generate two consecutive image of multiple moving objects
       (for now, objects can be a circle, triangle, or square)
       num_obj : int
       num_obj - number of objects
       img_height : int
       img_height - height of the image
       img_width : int
       img_width - width of the image
       speed_max : int
       speed_max - |x1 - x1'| < speed_max and |x2 - x2'| < speed_max
       size_max : int
       size_max : max size of objects
       return [img_prev, img_next]
       img_prev : np.ndarray
       img_prev - first image
       img_next : np.ndarray
       img_next - next image
    """
    img_prev = np.zeros([img_height, img_width], np.uint8)
    img_next = np.zeros([img_height, img_width], np.uint8)
    for i in range(num_obj):
        grey_value = int(255.0 * (float(i + 1) / float(num_obj)))
        x1 = np.random.choice(img_height, 1)[0]
        x2 = np.random.choice(img_width, 1)[0]
        size = np.random.choice(size_max, 1)[0] + 1
        while (True):
            x1_next = x1 + np.random.choice(speed_max * 2 + 1, 1)[0] - speed_max
            x2_next = x2 + np.random.choice(speed_max * 2 + 1, 1)[0] - speed_max
            if (x1_next >= 0 and x1_next < img_height and x2_next >= 0 and x2_next < img_width):
                break
        shape = np.random.choice(3, 1)[0]
        obj_size = 1 + np.random.choice(size_max, 1)[0]
        if (shape == 0):
            add_circle_to_image(img_prev, x1, x2, obj_size, grey_value)
            add_circle_to_image(img_next, x1_next, x2_next, obj_size, grey_value)
        elif (shape == 1):
            add_triangle_to_image(img_prev, x1, x2, obj_size, grey_value)
            add_triangle_to_image(img_next, x1_next, x2_next, obj_size, grey_value)
        else:
            add_square_to_image(img_prev, x1, x2, obj_size, grey_value)
            add_square_to_image(img_next, x1_next, x2_next, obj_size, grey_value)
    return [img_prev, img_next]
