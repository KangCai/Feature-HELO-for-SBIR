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
    Example of HELO extraction.
    :return:
    """
    # Extract
    img_file = '.\\images\\airplane_sketch.png'
    edge, image_ndarray, feature_filtered_helo = helo.HELO(img_file, is_sketch=True, draw=False)
    print feature_filtered_helo

def ExampleDistHELO():
    """
    Example of testing performance of HELO.
    :return:
    """
    helo_true = helo.HELO('.\\images\\airplane.png', is_sketch=False)
    print helo_true[-1]
    helo_false = helo.HELO('.\\images\\valve.png', is_sketch=False)
    print helo_false[-1]
    helo_query_sketch = helo.HELO('.\\images\\airplane_sketch.png', is_sketch=True)
    print helo_query_sketch[-1]
    print CalL1Distance(helo_query_sketch[-1],helo_true[-1])
    print CalL1Distance(helo_query_sketch[-1], helo_false[-1])

def CalL1Distance(feat1, feat2):
    """
    L1 distance (also named Manhattan distance)
    :param feat1: feature 1. 1-D array.
    :param feat2: feature 2. 1-D array.
    :return: int.
    """
    return numpy.sum(numpy.fabs(feat1 - feat2))

if __name__ == "__main__":
    # ExampleCannyEdgeDetection()
    ExampleHELO()
    # ExampleDistHELO()