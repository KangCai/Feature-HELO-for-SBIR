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

---

### 使用方法

---

### 算法介绍

**主干流程**

* 对于库中的图像,通过 Canny 边缘检测提取边缘图像;对于待问询的简笔画,直接进行二值化处理作为边缘图像;
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

假设已经使用了前文描述的旋转不变性，这里的翻转就很简单了，直接将特征反转即可。例如对于某个简笔画问询图，它经过了旋转不变性处理得到了
 5 维 HELO 特征，

```buildoutcfg
[1 2 3 4 5]
```

，在进行匹配时，将问询 HELO 特征翻转成

```buildoutcfg
[5 4 3 2 1]
```

，再与图像匹配库中进行匹配，对于任意一个图像，都选取两种特征 L1 距离中较小的值作为两者之间的距离。
