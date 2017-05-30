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

def optical_flow_for_fixed_scale(img_prev, img_next, poly_filter_size, win_size, covariance, iteration = 1, flow_input = None):
    """
        compute optical flow with Gunnar's algorithm on the original scale
        img_prev : np.array
        img_prev - previous image, 2d array
        img_next : np.array
        img_next - next image, 2d array
        poly_filter_size : int
        poly_filter_size - size for the filter used for fitting the polynomial
        win_size : int
        win_size - size of the neighbor for optial flow estimation
        covariance : array_like
        covariance - covariance matrix for the filter
        iteration : int
        iteration - number of iterations
        flow_input : np.ndarray
        flow_input - flow as prior knowledge, 3d array, shape (img.shape[0], img.shape[1], 2)
        return flow
        flow : np.ndarray
        flow - output optical flow, 3d array, shape (img.shape[0], img.shapep[1], 2))
    """
    if (flow_input is None):
        flow_prior = np.zero([img_prev.shape[0], img_prev.shape[1], 2], np.float32)
    else:
        flow_prior = flow_input
    A_1 = np.zeros([img_prev.shape[0], img_prev.shape[1], 2, 2])
    A_2 = np.zeros([img_prev.shape[0], img_prev.shape[1], 2, 2])
    B_1 = np.zeros([img_prev.shape[0], img_prev.shape[1], 2, 1])
    B_2 = np.zeros([img_prev.shape[0], img_prev.shape[1], 2, 1])
    for i in range(img_prev.shape[0]):
        for j in range(img_prev.shape[1]):
            w, x, y = collect_pixels(img_prev, i, j, poly_filter_size, covariance)
            a, b, c = polynomial_fit(w, x, y)
            A_1[i, j, :, :] = a
            B_1[i, j, :, :] = b
            w, x, y = collect_pixels(img_next, i, j, poly_filter_size, covariance)
            a, b, c = polynomial_fit(w, x, y)
            A_2[i, j, :, :] = a
            B_2[i, j, :, :] = b
    flow = np.zeros([img_prev.shape[0], img_prev.shape[1], 2], np.float32)
    for it in range(iteration):
        for i in range(img_prev.shape[0]):
            for j in range(img_prev.shape[1]):
                w = normal_filter(win_size, covariance)
                left_term = np.zeros([2, 2])
                right_term = np.zeros([2, 1])
                for x1 in range(i - win_size, i + win_size + 1):
                    for x2 in range(j - win_size, j + win_size + 1):
                        if (x1 >= 0 and x1 < img_prev.shape[0] and x2 >= 0 and x2 < img_prev.shape[1]):
                            x1_estimate = int(x1 + flow_prior[x1, x2, 0])
                            x2_estimate = int(x2 + flow_prior[x1, x2, 1])
                            if (x1_estimate >= 0 and x1_estimate < img_prev.shape[0] and x2_estimate >= 0 and x2_estimate < img_prev.shape[1]):
                                A = 0.5 * (A_1[i, j, :, :] + A_2[x1_estimate, x2_estimate, :, :])
                                A_T = A.transpose()
                                AD_prior = np.matmul(A, (flow_prior[x1, x2, :]).reshape([2, 1]))
                                delta_b = -0.5 * (B_2[x1_estimate, x2_estimate, :, :] - B_1[x1, x2, :, :]) + AD_prior
                                w_neighbor = w[x1 - (i - win_size), x2 - (j - win_size)]
                                left_term = left_term + w_neighbor * (np.matmul(A_T, A))
                                right_term = right_term + w_neighbor * (np.matmul(A_T, delta_b))
                d_x = np.matmul(linalg.inv(left_term), right_term)
                flow[i, j, :] = d_x[:, 0]
        flow_prior = flow.copy()
    return flow
