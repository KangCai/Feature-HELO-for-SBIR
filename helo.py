# coding=utf-8
"""
Desc: HELO feature extraction for sketch-based image retrieval.
Author: K. Cai.
Date: March 2019.
"""

import scipy.ndimage
import numpy
import canny

def HELO(img_file, is_sketch):
    # Get ndarray of img_file.
    ori_image_ndarray = scipy.ndimage.imread(img_file, flatten=True)
    # Preprocess.
    if is_sketch:
        # Simple thresholding.
        edge = 255 * (ori_image_ndarray > 0)
    else:
        # Canny edge detection.
        edge, _, _ = canny.Canny(img_file)
    # Divide the image into W * W blocks.
    image_blocks = _DivideImage(edge)
    # Sobel gradient.
    Gx, Gy = _BlockSobelGradient(image_blocks)
    # Calculate the denoised local orientation
    alpha_blocks = _BlockDenoisedOrientation(Gx, Gy)
    # K-bin histogram
    histogram_blocks = _Histogram(alpha_blocks)
    return edge, ori_image_ndarray, alpha_blocks, histogram_blocks

def _DivideImage(data, W=25):
    """
    Divide the image into W * W.
    :param data: ndarray.
    :param W: int.
    :return:
    """
    h, w = data.shape
    h_block_size, w_block_size = h / W, w / W
    image_blocks = numpy.reshape(data[:h_block_size * W, :w_block_size * W], (W, W, h_block_size, w_block_size))
    return image_blocks

def _BlockSobelGradient(data_blocks):
    """
    Sobel gradient.
    :param data_blocks: ndarray.
    :return: Sobel gradient for each block.
    """
    h_block_num, w_block_num, _, _ = data_blocks.shape
    Gx, Gy = numpy.zeros(data_blocks.shape), numpy.zeros(data_blocks.shape)
    for i in xrange(h_block_num):
        for j in xrange(w_block_num):
            block_data = data_blocks[i, j]
            Gx[i, j] = scipy.ndimage.filters.sobel(block_data, axis=1)
            Gy[i, j] = scipy.ndimage.filters.sobel(block_data, axis=0)
    return Gx, Gy

def _BlockDenoisedOrientation(Gx, Gy):
    """
    Minimize the noise sensitivity by a orientation local estimation. The main idea is to
    double the gradient angle and to square the gradient length. This has the effect that
    strong orientations have a higher vote in the local average orientation than weaker
    orientations.
    :param Gx: ndarray. shape: (W, W, None, None).
    :param Gx: ndarray. shape: (W, W, None, None).
    :return: Local orientation
    """
    # Double the gradient angle and to square the gradient length
    h_block_num, w_block_num, _, _ = Gx.shape
    Lx, Ly = numpy.zeros((h_block_num, w_block_num)), numpy.zeros((h_block_num, w_block_num))
    for i in xrange(h_block_num):
        for j in xrange(w_block_num):
            Ly[i, j] = numpy.sum(Gx[i, j] * Gy[i, j])
            Lx[i, j] = numpy.sum(Gx[i, j]**2 - Gy[i, j]**2)
    # Gaussian filter
    sigma, window_size = 0.5, 3
    truncate = (((window_size - 1) / 2) - 0.5) / sigma
    Lx = scipy.ndimage.filters.gaussian_filter(Lx, sigma=sigma, truncate=truncate)
    Ly = scipy.ndimage.filters.gaussian_filter(Lx, sigma=sigma, truncate=truncate)
    # Calculate the local orientation for block
    alpha = 0.5 * (numpy.arctan2(Ly, Lx) + numpy.pi)
    return alpha

def _Histogram(alpha_blocks, K=72):
    """
    Histogram of local orientation
    :param alpha_blocks: ndarray. shape: (W, W).
    :param K: int. Number of histogram bin, default is 72.
    :return:
    """
    hist_blocks = alpha_blocks / (numpy.pi / K)
    hist_blocks = hist_blocks.astype(numpy.int32) * numpy.pi / K
    return hist_blocks