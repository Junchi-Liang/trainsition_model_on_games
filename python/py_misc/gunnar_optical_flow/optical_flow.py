import cv2
import numpy as np
import scipy.misc
import scipy.stats
import numpy.linalg as linalg

def normal_filter(size, covariance):
    """
        size : int
        size - the size of the filter
        covariance : array_like
        covariance - Covariance matrix of the distribution
        return w
        w : np.array
        w - filter from normal distibution, its shape should be (2 * size + 1, 2 * size + 1)
    """
    w = np.zeros([1 + 2 * size, 1 + 2 * size])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            w[i, j] = scipy.stats.multivariate_normal.pdf(np.asarray([i - size, j - size]), mean = np.asarray([0, 0]), cov = covariance)
    return w

def polynomial_fit(w, x, y):
    """
        fit polynomial (x^T)Ax + (B^T)x + C
        w : np.array
        w - weights for each sample, 1d array
        x : np.array
        x - positions of pixels, its shape should be (2, n)
        y : np.array
        y - intensities of each pixels, 1d array
        return [A, B, C]
        A : np.array
        A - coefficient, shape (2, 2)
        B: np.array
        B - coefficient, shape (2, 1)
        C : float
        C - coefficient
    """
    n = x.shape[1]
    w_matrix = np.zeros([n, n], np.float32)
    x_matrix = np.zeros([6, n], np.float32)
    y_vector = np.float32(y.reshape([n, 1]))
    for i in range(n):
        w_matrix[i, i] = w[i]
        x1, x2 = x[:, i]
        x_matrix[:, i] = [x1 * x1, x1 * x2, x2 * x2, x1, x2, 1.0]
    x_T = x_matrix.transpose()
    right_side = np.matmul(x_T, np.matmul(w_matrix, y_vector))
    left_side = np.matmul(x_T, np.matmul(w_matrix, x_matrix))
    beta = np.matmul(linalg.inv(left_side), right_side)
    a_11 = beta[0, 0]
    a_12 = beta[1, 0] / 2.0
    a_22 = beta[2, 0]
    A = np.asarray([[a_11, a_12], [a_12, a_22]], np.float32)
    B = np.asarray([[beta[3, 0]], [beta[4, 0]]], np.float32)
    C = beta[5, 0]
    return [A, B, C]

def collect_pixels(img, p_x, p_y, size, covariance):
    """
        form dataset for polynomial fit
        img : np.array
        img - input image, 2d array
        p_x : int
        p_x - location of the center pixel
        p_y : int
        p_y - location of the center pixel
        size : int
        size - the size for the filter
        covariance : array_like
        covariance - covariance matrix for the filter
        return [w, x, y]
        w : np.array
        w - weights for each sample, 1d array
        x : np.array
        x - positions of pixels, its shape should be (2, n)
        y : np.array
        y - intensities of each pixels, 1d array       
    """
    w_filter = normal_filter(size, covariance)
    x_list = []
    y_list = []
    w_list = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            x1 = p_x + i
            x2 = p_y + j
            if (x1 >= 0 and x1 < img.shape[0] and x2 >= 0 and x2 < img.shape[1]):
                x_list.append([x1, x2])
                y_list.append(img[x1, x2])
                w_list.append(w_filter[i + size, j + size])
    x = (np.asarray(x_list, np.float32)).transpose()
    w = np.asarray(w_list, np.float32)
    y = np.asarray(y_list, np.float32)
    return [w, x, y]
