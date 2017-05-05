fcn-tensorflow的学习

参考：https://github.com/MarvinTeichmann/tensorflow-fcn

步骤：
1. 把label的像素存到txt中，每个txt都对应一个原始图像的标签
2. 把原始图像名称存到一个txt中
3. 代码处理3通道，如果是灰度图像，需要通道叠加成3通道

util.py------着色
