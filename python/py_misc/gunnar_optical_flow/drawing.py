import cv2
import numpy as np
import scipy.misc
import numpy.random

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
            if (i ** 2 + j ** 2 < r ** 2):
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

def random_multiple_moving_objects(num_obj, speed_max, size_max, img_height, img_width, noise_in_prev = False, std_in_prev = None, noise_in_next = False, std_in_next = None):
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
       noise_in_prev : boolean
       noise_in_prev - if the previous image should contain noise
       std_in_prev : None or float
       std_in_prev - standard deviation of noise in previous image
       noise_in_next : boolean
       noise_in_next - if the next image should contain noise
       std_in_next : None of float
       std_in_next - standard deviation of noise in next image
       return [img_prev, img_next]
       img_prev : np.ndarray
       img_prev - first image
       img_next : np.ndarray
       img_next - next image
       motion_field : np.ndarray
       motion_field - 3d array, shape (img_height, img_width, 2), motion field
       model : np.ndarray
       model - model for the whole image, 3d array, shape (img_height, img_width, 5):
               model[i, j, 0..1] = location of the object to which this pixel belongs
               model[i, j, 2] = shape of the object to which this pixel belongs
               model[i, j, 3] = size of the object to which this pixel belongs
               model[i, j, 4] = greyscale value of the object to which this pixel belongs
    """
    img_prev = np.zeros([img_height, img_width], np.uint8)
    img_next = np.zeros([img_height, img_width], np.uint8)
    motion_field = np.zeros([img_height, img_width, 2])
    model = np.zeros([img_height, img_width, 5])
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
        obj_size = int(1 + np.random.choice(size_max, 1)[0])
        if (shape == 0):
            add_circle_to_image(img_prev, x1, x2, obj_size, grey_value)
            add_circle_to_image(img_next, x1_next, x2_next, obj_size, grey_value)
        elif (shape == 1):
            add_triangle_to_image(img_prev, x1, x2, obj_size, grey_value)
            add_triangle_to_image(img_next, x1_next, x2_next, obj_size, grey_value)
        else:
            add_square_to_image(img_prev, x1, x2, obj_size, grey_value)
            add_square_to_image(img_next, x1_next, x2_next, obj_size, grey_value)
        for p1 in range(x1 - obj_size, x1 + obj_size + 1):
            for p2 in range(x2 - obj_size, x2 + obj_size + 1):
                if (p1 >= 0 and p1 < img_height and p2 >= 0 and p2 < img_width and img_prev[p1, p2] == grey_value):
                    motion_field[p1, p2, 0] = x1_next - x1
                    motion_field[p1, p2, 1] = x2_next - x2
                    model[p1, p2, 0] = x1
                    model[p1, p2, 1] = x2
                    model[p1, p2, 2] = shape
                    model[p1, p2, 3] = obj_size
                    model[p1, p2, 4] = grey_value
    if (noise_in_prev):
        for i in range(img_height):
            for j in range(img_width):
                noise = numpy.random.normal(scale = std_in_prev)
                v = min(max(img_prev[i, j] + noise, 0), 255)
                img_prev[i, j] = int(v)
    if (noise_in_next):
        for i in range(img_height):
            for j in range(img_width):
                noise = numpy.random.normal(scale = std_in_next)
                v = min(max(img_next[i, j] + noise, 0), 255)
                img_next[i, j] = int(v)
    return [img_prev, img_next, motion_field, model]

def average_optical_flow_from_motion_field(img_prev, img_next, motion_field):
    """
        average optical flow (least square solution) given matching between pixels in previous image and pixels in next image
        img_prev : np.array
        img_prev - 2d array, previous image
        img_next : np.array
        img_next - 2d array, next image
        motion_field : np.ndarray
        motion_field - 3d array, shape (img_prev.shape[0], img_prev.shape[1], 2), motion field
        return flow
        flow : np.ndarray
        flow - 3d array, shape (img_prev.shape[0], img_prev.shape[1], 2), result optical flow
    """
    flow_for_segment = {}
    cnt_for_segment = {}
    for i in range(img_prev.shape[0]):
        for j in range(img_prev.shape[1]):
            x1 = int(i + motion_field[i, j , 0])
            x2 = int(j + motion_field[i, j , 1])
            if (x1 >= 0 and x1 < img_next.shape[0] and x2 >= 0 and x2 < img_next.shape[1] and img_prev[i, j] > 0 and img_prev[i, j] == img_next[x1, x2]):
                if (img_prev[i, j] in flow_for_segment):
                    flow_for_segment[img_prev[i, j]][0] = flow_for_segment[img_prev[i, j]][0] + motion_field[i, j, 0]
                    flow_for_segment[img_prev[i, j]][1] = flow_for_segment[img_prev[i, j]][1] + motion_field[i, j, 1]
                    cnt_for_segment[img_prev[i, j]] = 1 + cnt_for_segment[img_prev[i, j]]
                else:
                    cnt_for_segment[img_prev[i, j]] = 1
                    flow_for_segment[img_prev[i, j]] = [motion_field[i, j, 0], motion_field[i, j, 1]]
    flow = np.zeros([img_prev.shape[0], img_prev.shape[1], 2])
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            if (img_prev[i, j] > 0 and img_prev[i, j] in cnt_for_segment):
                flow[i, j, 0] = float(flow_for_segment[img_prev[i, j]][0]) / float(cnt_for_segment[img_prev[i, j]])
                flow[i, j, 1] = float(flow_for_segment[img_prev[i, j]][1]) / float(cnt_for_segment[img_prev[i, j]])
    return flow

def reconstruct_from_flow(img_prev, flow, model = None):
    """
        predict the next frame given previous image and the optical flow or motion field
        img_prev : np.array
        img_prev - 2d array, previous frame
        flow : np.ndarray
        flow - 3d array, shape (img_prev.shape[0], img_prev.shape[1], 2), optical flow or motion field
        model : np.ndarray
        model - model for the whole image, 3d array, shape (img_height, img_width, 5):
                model[i, j, 0..1] = location of the object to which this pixel belongs
                model[i, j, 2] = shape of the object to which this pixel belongs
                model[i, j, 3] = size of the object to which this pixel belongs
                model[i, j, 4] = greyscale value of the object to which this pixel belongs
        return img_next
        img_next : np.array
        img_next - 2d array, predicted next frame
    """
    img_next = np.zeros(img_prev.shape, np.uint8)
    if (model is None):
        for i in range(img_prev.shape[0]):
            for j in range(img_prev.shape[1]):
                x1 = int(i + flow[i, j, 0])
                x2 = int(j + flow[i, j, 1])
                if (img_prev[i, j] > 0 and x1 >= 0 and x1 < img_prev.shape[0] and x2 >= 0 and x2 < img_prev.shape[1]):
                    img_next[x1, x2] = img_prev[i, j]
    else:
        segment_pos = {}
        for i in range(img_prev.shape[0]):
            for j in range(img_prev.shape[1]):
                if (model[i, j, 4] > 0 and (model[i, j, 4] not in segment_pos) and img_prev[i, j] == model[i, j, 4]):
                    x1 = int(model[i, j, 0] + flow[i, j, 0])
                    x2 = int(model[i, j, 1] + flow[i, j, 1])
                    segment_pos[model[i, j, 4]] = [x1, x2]
                    if (model[i, j, 2] == 0):
                        add_circle_to_image(img_next, x1, x2, int(model[i, j, 3]), model[i, j, 4])
                    elif (model[i, j, 2] == 1):
                        add_triangle_to_image(img_next, x1, x2, int(model[i, j, 3]), model[i, j, 4])
                    else:
                        add_square_to_image(img_next, x1, x2, int(model[i, j, 3]), model[i, j, 4])
    return img_next
