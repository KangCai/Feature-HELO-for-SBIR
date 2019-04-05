# Feature HELO for SBIR

[English](https://github.com/KangCai/An-Improved-Histogram-of-Edge-Local-Orientations-for-Sketch-Based-Image-Retrieval/blob/master/README.md)

很高兴你发现了这个项目，该项目是一个论文方法实现工程，原论文是 "An Improved Histogram of Edge Local Orientations for 
Sketch-Based Image Retrieval"。有任何意见或建议欢迎提出。

---

### 运行环境

* Python 2
* numpy
* scipy
* pylab

---

### 注意

* Canny 边缘检测算法: canny.py
* HELO 特征提取: helo.py (依赖 canny.py)
* 使用示例: examples.py (依赖 canny.py 和 helo.py)

---

### 使用方法

Canny 边缘检测:

```buildoutcfg
img_file = '.\\images\\airplane.png'
edge, image_ndarray, nms = canny.Canny(img_file)
```

HELO 特征提取:

```buildoutcfg
img_file = '.\\images\\airplane.png'
edge, image_ndarray, alpha_blocks, helo = helo.HELO(img_file, is_sketch=False, draw=False, calc_flip=True)
# If calc_flip == True
ori_helo, flip_helo = helo
# If calc_flip == False
# ori_helo = helo
```

更多使用细节可参考 examples.py.

---

### 算法介绍

**主干流程**

* 对于库中的图像,通过 Canny 边缘检测提取边缘图像;对于待问询的简笔画,直接进行二值化处理作为边缘图像;
* 用一个最小的矩形框住边缘图所有非0的像素;
* 将边缘图像均匀划分为 W * W 个子图像块;
* 通过 Sobel 算子计算每个子图像块沿 x 方向和 y 方向的梯度;
* 根据子图像块中每个像素的梯度,计算子图像的局部方向;
* 将连续值形式的子图像局部方向进行离散化;
* 用词袋模型(Bag-of-words model)统计所有子图像块离散化后的局部方向,其中忽略掉边缘像素点数目低于一定阈值的子图像块,作为最终特征.

**旋转不变性**

共采用如下3种方法，

1. 梯度减去极坐标角度
2. 梯度减去 PCA 第1梯度特征向量
3. 方法1和方法2结果的加权平均

在我看来,方法1无论在可解释性和鲁棒性方面都优于方法2,但完事无绝对.我的建议是选择适合自己应用项目的方法。

**翻转不变性**

假设我们后续会使用了前文描述的旋转不变性，这里的翻转就很简单了，直接将 alpha 反转即可。例如对于某个简笔画问询图，它提取出了每个块的 alpha 值
 3 * 3 的 alpha blocks，

```buildoutcfg
[1 2 3,
4 5 6,
7 8 9]
```

，将其翻转成

```buildoutcfg
[3 2 1,
6 5 4,
9 8 7]
```

，再进行上文的旋转不变性变换,提取出 HELO 特征,这样一来,我们就为每个图像生成了两个 HELO 特征.将其与图像匹配库中的特征进行匹配，对于任意一个图像，都选取两种特征 L1 距离中较小的值作为两者之间的距离。
