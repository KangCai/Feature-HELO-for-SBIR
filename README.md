# Feature HELO for SBIR

[中文](https://github.com/KangCai/An-Improved-Histogram-of-Edge-Local-Orientations-for-Sketch-Based-Image-Retrieval/blob/master/README_zh_CN.md)

Kang is glad you find me. This is an project for realization of paper "An Improved Histogram of Edge Local 
Orientations for Sketch-Based Image Retrieval". Any suggestions or comments well be welcome.

---

### Environment

* Python 2
* numpy
* scipy
* pylab

---

### Note

---

### How to use

---

### Module

**Main Procedure**

* For test images in database, use the Canny algorithm to get an edge map. For sketch, use a simple thresholding to get a binary representation.
* Divide the image into W × W blocks.
* Compute gradient respect to x and to y for each pixel in a block by applying Sobel masks.
* Compute local orientations of each block.
* Create a K-bin histogram to represent the distribution of the local orientation.
* Map each local orientation αij to the corresponding histogram bin to
increase it by one. Blocks with a few edge points are neglected. And the final representation of histogram is called the histogram of edge local orientation (HELO).

**Rotational invariance**

Realize by applying three methods as follows：

1. Compute a 2-d eigenvector v representing the axis
with higher variance of the pixel (with value 1) distribution using PCA. Normalize both the sketch and the test image abstract representation by rotating them −α degrees around their center of mass, in which α is the orientation of v.
2. Transform both the test image and the sketch abstract representation into polar coordinates. Normalize both the the local orientations of each block sketch and the test image by rotating the corresponding polar angle.
3. The weighted mean of method 1 & 2.

In my opinion, method 2 is more reasonable and robust than method 1. But it is not always the case. A suggestion is that you should choose a proper method for your own application.

**Flip invariance**

Supposing we have used the method of rotational invariance described above, the realization of flip invariance is very easy then. We just need reverse the HELO feature. For example, we get a 5-d HELO feature of a query sketch as follows,

```buildoutcfg
[1 2 3 4 5]
```

When we match it with the HELO features of database, we also need to reverse the feature into

```buildoutcfg
[5 4 3 2 1]
```

，then match them again. For each image-sketch pair for matching, we always choose the minimum value of two L1 distance as the final distance of matching pair.
