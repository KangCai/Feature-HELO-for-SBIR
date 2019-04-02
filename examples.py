# coding=utf-8
"""
Desc: Examples for Canny, HELO.
Author: K. Cai.
Date: March 2019.
"""

import pylab
import canny
import helo
import numpy

def ExampleCannyEdgeDetection():
    """
    Example of canny edge detection.
    :return:
    """
    # Extract
    img_file = '.\\images\\airplane.png'
    edge, image_ndarray, nms = canny.Canny(img_file)
    # Draw
    pylab.title('input image')
    pylab.draw()
    pylab.imshow(image_ndarray)
    if len(image_ndarray.shape) == 2:
        pylab.gray()
    pylab.figure()
    pylab.imshow(nms)
    pylab.figure()
    pylab.imshow(edge)
    pylab.show()

def ExampleHELO():
    """

    :return:
    """
    # img_file = '.\\images\\1006_guishihei\\100602.jpg'
    img_file = '.\\images\\1019_guanhu\\101905.jpg'
    edge, image_ndarray, alpha_blocks, feature_filtered_helo = helo.HELO(img_file, is_sketch=False, draw=True, calc_flip=False)
    print feature_filtered_helo

def ExampleHELOSketch():
    """
    Example of HELO extraction.
    :return:
    """
    # Extract
    img_file = '.\\images\\airplane_sketch.png'
    edge, image_ndarray, alpha_blocks, feature_filtered_helo = helo.HELO(img_file, is_sketch=True, draw=False, calc_flip=True)
    print feature_filtered_helo

def ExampleDistHELO():
    """
    Example of testing performance of HELO.
    :return:
    """
    for rotate_type in  ('RAW', 'PCA', 'PC', 'R'):
        helo_true = helo.HELO('.\\images\\airplane.png', is_sketch=False, rotate_type=rotate_type)
        print helo_true[-1]
        helo_false = helo.HELO('.\\images\\valve.png', is_sketch=False, rotate_type=rotate_type)
        print helo_false[-1]
        helo_query_sketch = helo.HELO('.\\images\\airplane_sketch.png', is_sketch=True, rotate_type=rotate_type)
        print helo_query_sketch[-1]
        print CalL1Distance(helo_query_sketch[-1],helo_true[-1])
        print CalL1Distance(helo_query_sketch[-1], helo_false[-1])
        print '=' * 10

def ExampleDistHELO_PAJ_3_types():
    """
    Example of testing performance of HELO for PAJ with 3 types.
    :return:
    """
    rotate_type = 'R'
    gsh_fp_list = ['1006_guishihei\\1006.jpg', '1006_guishihei\\100601.jpg', '1006_guishihei\\100602.jpg',
            '1006_guishihei\\100603.jpg']
    gsh_helo_list = []
    for gsh_fp in gsh_fp_list:
        gsh_helo_list.append(helo.HELO('.\\images\\' + gsh_fp, is_sketch=False, calc_flip=True, rotate_type=rotate_type)[-1])
    gh_fp_list = ['1019_guanhu\\1019.jpg', '1019_guanhu\\101901.jpg', '1019_guanhu\\101902.jpg',
                   '1019_guanhu\\101903.jpg', '1019_guanhu\\101904.jpg', '1019_guanhu\\101905.jpg']
    gh_helo_list = []
    for gh_fp in gh_fp_list:
        gh_helo_list.append(helo.HELO('.\\images\\' + gh_fp, is_sketch=False, calc_flip=True, rotate_type=rotate_type)[-1])
    gsh_guanhu_helo_list = []
    gsh_guanhu_helo_list.extend(gsh_helo_list)
    gsh_guanhu_helo_list.extend(gh_helo_list)
    for feat_1 in gsh_guanhu_helo_list:
        for feat_2 in gsh_guanhu_helo_list:
            print CalL1DistancePair(feat_1[0:2], feat_2[0]),
        print
    print '-' * 10
    for feat_1 in gsh_guanhu_helo_list:
        for feat_2 in gsh_guanhu_helo_list:
            print CalL1Distance(feat_1[0], feat_2[0], consider_reverse=True),
        print

def CalL1DistancePair(feat1_pair, feat2):
    """

    :param feat1_pair:
    :param feat2:
    :return:
    """
    d1 = numpy.sum(numpy.fabs(feat1_pair[0] - feat2))
    d2 = numpy.sum(numpy.fabs(feat1_pair[1] - feat2))
    return min(d1, d2)

def CalL1Distance(feat1, feat2, consider_reverse=False):
    """
    L1 distance (also named Manhattan distance)
    :param feat1: feature 1. 1-D array.
    :param feat2: feature 2. 1-D array.
    :param consider_reverse: feature 2. 1-D array.
    :return: int.
    """
    d1 = numpy.sum(numpy.fabs(feat1 - feat2))
    d2 = numpy.sum(numpy.fabs(helo.FlipInvariance(feat1) - feat2)) if consider_reverse else 9999999
    return min(d1, d2)

if __name__ == "__main__":
    # ExampleCannyEdgeDetection()
    ExampleDistHELO_PAJ_3_types()
    # ExampleHELO()
    # ExampleDistHELO()