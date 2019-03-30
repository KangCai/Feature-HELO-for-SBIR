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



**旋转不变性**

共采用如下3种方法，

* 梯度减去极坐标角度
* 梯度减去PCA第1梯度特征向量
* 加权平均

建议选择其中对自己项目效果好的方法。

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
