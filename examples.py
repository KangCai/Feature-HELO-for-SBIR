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
    img_file = '.\\images\\airplane.png'
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
        # print helo_true[-1]
        helo_false = helo.HELO('.\\images\\valve.png', is_sketch=False, rotate_type=rotate_type)
        # print helo_false[-1]
        helo_query_sketch = helo.HELO('.\\images\\airplane_sketch.png', is_sketch=True, rotate_type=rotate_type, calc_flip=True)
        # print helo_query_sketch[-1]
        print '=' * 10, rotate_type, '=' * 10
        print 'True pair:', CalL1DistancePair(helo_query_sketch[-1],helo_true[-1])
        print 'Fakse pair:', CalL1DistancePair(helo_query_sketch[-1], helo_false[-1])


def CalL1DistancePair(feat1_pair, feat2):
    """

    :param feat1_pair:
    :param feat2:
    :return:
    """
    d1 = numpy.sum(numpy.fabs(feat1_pair[0] - feat2))
    d2 = numpy.sum(numpy.fabs(feat1_pair[1] - feat2))
    return min(d1, d2)

if __name__ == "__main__":
    # ExampleCannyEdgeDetection()
    # ExampleHELO()
    # ExampleHELOSketch()
    ExampleDistHELO()