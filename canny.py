# coding=utf-8
"""
Desc: canny edge detection.
Author: K. Cai.
Date: March 2019.
"""

import scipy.ndimage
import numpy

def Canny(img_file, threshold_low=120, threshold_high=180, binary_edge=True):
    """
    Canny edge detection.
    :param img_file:
    :param threshold_low:
    :param threshold_high:
    :param binary_edge:
    :return:
    """
    # Get ndarray of img_file
    ori_image_ndarray = scipy.ndimage.imread(img_file, flatten=True)
    # Gassian filter
    image_ndarray = _GaussianFilter(ori_image_ndarray)
    # Format pixel value
    image_ndarray = _FormatPixelValue(image_ndarray)
    # Sobel gradient
    Gx, Gy = _SobelGradient(image_ndarray)
    # Gray gradiant
    Gm, Gd = _GrayGradient(Gx, Gy)
    # Non-maximum suppression, NMS
    nms = _NonMaximumSuppression(Gm, Gd)
    # Double Thresholding
    edge = _DoubleThresholding(nms, threshold_low, threshold_high, binary_edge)
    return edge, ori_image_ndarray, nms

def _GaussianFilter(data, sigma=1.5, window_size=9):
    """
    Gaussian filter.
    :param data: ndarray.
    :param sigma:
    :param window_size:
    :return:
    """
    truncate = (((window_size - 1) / 2) - 0.5) / sigma
    return scipy.ndimage.filters.gaussian_filter(data, sigma=sigma, truncate=truncate)

def _FormatPixelValue(data):
    """
    Format pixel value.
    :param data: ndarray.
    :return:
    """
    data_max = data.max()
    if data_max > 255:
        data *= 255.0 / 65535
    return data

def _SobelGradient(data):
    """
    Sobel gradient.
    :param data:
    :return: Sobel gradient.
    """
    Gx = scipy.ndimage.filters.sobel(data, axis=1)
    Gy = scipy.ndimage.filters.sobel(data, axis=0)
    return Gx, Gy

def _GrayGradient(Gx, Gy):
    """
    Canny gray gradient.
    :param Gx: ndarray.
    :param Gy: ndarray.
    :return: Gradient magnitudes and directions of a grayscale image, Gd: [-pi/2, +pi/2]
    """
    Gm = numpy.sqrt(Gx**2 + Gy**2)
    Gd = numpy.arctan2(Gy, Gx)
    Gd[Gd > 0.5 * numpy.pi] -= numpy.pi
    Gd[Gd < -0.5 * numpy.pi] += numpy.pi
    return Gm, Gd

def _NonMaximumSuppression(Gm, Gd, threshold=1.0):
    """
    Non-maximum suppression.
    :param Gm: gradient magnitudes
    :param Gd: gradient directions, -pi/2 to +pi/2
    :param threshold:
    :return: ndarray. Gradient magnitude if local max, 0 otherwise
    """
    nms = numpy.zeros(Gm.shape, Gm.dtype)
    h, w = Gm.shape
    for x in xrange(1, w - 1):
        for y in xrange(1, h - 1):
            mag = Gm[y, x]
            if mag < threshold:
                continue
            theta = Gd[y, x]
            # abs(teta) > 1.1781: teta < -67.5 degrees or teta > 67.5 degrees
            dx, dy = 0, -1
            # -22.5 < theta < 22.5 degrees
            if numpy.abs(theta) <= 0.3927:
                dx, dy = 1, 0
            # 22.5 < theta < 67.5 degrees
            elif 0.3927 < theta < 1.1781:
                dx, dy = 1, 1
            # -67.5 < theta < -22.5 degrees
            elif -1.1781 < theta < -0.3927:
                dx, dy = 1, -1
            if mag > Gm[y+dy, x+dx] and mag > Gm[y-dy, x-dx]:
                nms[y, x] = mag
    return nms

def _DoubleThresholding(data, thLow, thHigh, binary_edge):
    """
    Double thresholding.
    :param data: ndarray.
    :param thLow: low threshold.
    :param thHigh: High threshold.
    :param binary_edge: binary edge.
    :return: ndarray.
    """
    labels, n = scipy.ndimage.measurements.label(data > thLow, structure=numpy.ones((3, 3)))
    for i in range(1, n):
        upper = numpy.amax(data[labels==i])
        if upper < thHigh:
            labels[labels==i] = 0
    return 255*(labels>0) if binary_edge else data*(labels>0)