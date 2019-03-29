# coding=utf-8
"""
Desc: Examples for Canny, HELO.
Author: K. Cai.
Date: March 2019.
"""

import pylab
import canny
import helo

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
    img_file = '.\\images\\airplane.png'
    edge, image_ndarray, alpha, histogram, filtered_histogram, feature_helo, feature_filtered_helo = \
            helo.HELO(img_file, is_sketch=False)
    print feature_helo
    print feature_filtered_helo
    # Draw
    pylab.title('input image')
    pylab.draw()
    if len(image_ndarray.shape) == 2:
        pylab.gray()
    pylab.imshow(image_ndarray)
    pylab.figure()
    pylab.imshow(edge)
    pylab.figure()
    pylab.imshow(alpha)
    helo.DrawHELO(alpha)
    helo.DrawHELO(histogram)
    helo.DrawHELO(filtered_histogram)
    pylab.show()


if __name__ == "__main__":
    # ExampleCannyEdgeDetection()
    ExampleHELO()