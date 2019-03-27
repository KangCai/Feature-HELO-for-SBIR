# coding=utf-8

import canny
import scipy.misc
import pylab

def ExampleCannyEdgeDetection():
    """
    Example of canny edge detection.
    :return:
    """
    # Extract
    img_file = '.\\images\\valve.png'
    edge, image_ndarray, nms = canny.Canny(img_file)
    # Draw
    pylab.title('input image')
    pylab.draw()
    if len(image_ndarray.shape) == 2:
        pylab.gray()
    pylab.figure()
    pylab.imshow(nms)
    pylab.figure()
    pylab.imshow(edge)
    pylab.show()

if __name__ == "__main__":
    ExampleCannyEdgeDetection()