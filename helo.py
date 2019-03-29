# coding=utf-8
"""
Desc: HELO feature extraction for sketch-based image retrieval.
Author: K. Cai.
Date: March 2019.
"""

import scipy.ndimage
import numpy
import canny

def HELO(img_file, is_sketch, W=25, K=72, th_edge_ratio=0.5):
    """
    Extract HELO feature.
    :param img_file:
    :param is_sketch:
    :param W:
    :param K:
    :param th_edge_ratio:
    :return:
    """
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
    image_blocks = _DivideImage(edge, W)
    # Sobel gradient.
    Gx, Gy = _BlockSobelGradient(image_blocks)
    # Calculate the denoised local orientation
    alpha_blocks = _BlockDenoisedOrientation(Gx, Gy)
    # K-bin histogram
    histogram_blocks = _Histogram(alpha_blocks, K)
    # Filter blocks by removing the block with a few edge points.
    filtered_histogram_blocks = _FilterBlocks(histogram_blocks, image_blocks, th_edge_ratio)
    # Extract histogram feature
    feature_helo, feature_filtered_helo = _ExtractHistFeature(K, alpha_blocks, image_blocks, th_edge_ratio)
    return edge, ori_image_ndarray, alpha_blocks, histogram_blocks, filtered_histogram_blocks, feature_helo, feature_filtered_helo

def DrawHELO(histogram_blocks):
    """
    Draw the orientation field of Fig.1 in the paper.
    :param histogram_blocks:
    :return:
    """
    import pylab
    pylab.figure()
    h_block_num, w_block_num = histogram_blocks.shape
    for i in xrange(h_block_num):
        for j in xrange(w_block_num):
            radian = histogram_blocks[i, j]
            base_x, base_y = j + 0.5, h_block_num - i - 0.5
            if radian <= numpy.pi * 0.25 or radian >= numpy.pi * 0.75:
                dx1, dx2 = 1, -1
                dy1, dy2 = -numpy.tan(radian), numpy.tan(radian)
            elif radian == numpy.pi * 0.5:
                dx1, dx2 = 0, 0
                dy1, dy2 = -1, 1
            else:
                dx1, dx2 = 1.0 / numpy.tan(radian), -1.0 / numpy.tan(radian)
                dy1, dy2 = -1, 1
            x_pair = [base_x + 0.5 * dx1, base_x + 0.5 * dx2]
            y_pair = [base_y + 0.5 * dy1, base_y + 0.5 * dy2]
            pylab.plot(x_pair, y_pair, color='blue')

def _DivideImage(data, W):
    """
    Divide the image into W * W.
    :param data: ndarray.
    :param W: int.
    :return:
    """
    h, w = data.shape
    h_block_size, w_block_size = h / W, w / W
    image_blocks = numpy.zeros((W, W, h_block_size, w_block_size))
    for i in xrange(W):
        for j in xrange(W):
            image_blocks[i, j, :, :] = data[h_block_size*i:h_block_size*(i+1), w_block_size*j:w_block_size*(j+1)]
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
            Ly[i, j] = 2 * numpy.sum(Gx[i, j] * Gy[i, j])
            Lx[i, j] = numpy.sum(Gx[i, j]**2 - Gy[i, j]**2)
    # Gaussian filter
    sigma, window_size = 0.5, 3
    truncate = (((window_size - 1) / 2) - 0.5) / sigma
    Lx = scipy.ndimage.filters.gaussian_filter(Lx, sigma=sigma, truncate=truncate)
    Ly = scipy.ndimage.filters.gaussian_filter(Ly, sigma=sigma, truncate=truncate)
    # Calculate the local orientation for block
    alpha = 0.5 * (numpy.arctan2(Ly, Lx) + numpy.pi)
    return alpha

def _Histogram(alpha_blocks, K):
    """
    Histogram of local orientation
    :param alpha_blocks: ndarray. shape: (W, W).
    :param K: int. Number of histogram bin, default is 72.
    :return:
    """
    hist_blocks = alpha_blocks / (numpy.pi / K)
    hist_blocks = hist_blocks.astype(numpy.int32) * numpy.pi / K
    return hist_blocks

def _FilterBlocks(histogram, image_blocks, threshold_edge_ratio):
    """
    Filter blocks by removing the block with a few edge points.
    :param histogram:
    :param image_blocks:
    :param threshold_edge_ratio:
    :return:
    """
    h_block_num, w_block_num, h_block_size, w_block_size = image_blocks.shape
    threshold_edge = threshold_edge_ratio * numpy.max((h_block_size, w_block_size))
    filtered_histogram = numpy.zeros(histogram.shape)
    for i in xrange(h_block_num):
        for j in xrange(w_block_num):
            image_block = image_blocks[i, j]
            if len(image_block[image_block!= 0]) < threshold_edge:
                filtered_histogram[i, j] = numpy.pi / 2.0
            else:
                filtered_histogram[i, j] = histogram[i, j]
    return filtered_histogram

def _ExtractHistFeature(K, alpha_blocks, image_blocks, th_edge_ratio):
    """
    Extract histogram feature.
    :param alpha_blocks: ndarray. shape: (W, W).
    :param image_blocks: ndarray. shape: (W, W, None, None).
    :param th_edge_ratio: threshould ratio of edge.
    :return: histogram feature, filtered_hist.
    """
    h_block_num, w_block_num = alpha_blocks.shape
    _, _, h_block_size, w_block_size = image_blocks.shape
    threshold_edge = th_edge_ratio * numpy.max((h_block_size, w_block_size))
    feature_hist, feature_filtered_hist = numpy.zeros(K), numpy.zeros(K)
    for i in xrange(h_block_num):
        for j in xrange(w_block_num):
            alpha = alpha_blocks[i, j]
            alpha_bin_idx = int(alpha / (numpy.pi / K))
            # Without filter
            feature_hist[alpha_bin_idx] += 1
            # With filter
            image_block = image_blocks[i, j]
            if len(image_block[image_block != 0]) >= threshold_edge:
                feature_filtered_hist[alpha_bin_idx] += 1
    return feature_hist, feature_filtered_hist