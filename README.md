# global-and-local-consistent-image-completion-base-on-matconvnet

## 说明
使用`matconvnet`实现`《Globally and locally consistent image completion》`

需要先安装`matconvnet`，[matconvnet homepage](http://www.vlfeat.org/matconvnet/)，本项目在`matconvnet 1.0-beta25`上搭建

安装好`matconvnet`之后，将本项目中的文件复制到`matconvnet`对应的目录下

训练的入口函数是`glcic_train()`，直接运行`glcic_train.m`开始训练

注意在训练过程中会保存当前训练的部分修复结果，即对训练数据集图像的修复结果，作为参考使用，该代码在`process_epoch`中

对于项目根目录下的文件，可以先在`example`文件夹中创建一个新文件夹，再将文件复制到其中
## 参考
1. `Ishikawa H, Ishikawa H, Ishikawa H. Globally and locally consistent image completion[M]. ACM, 2017.`
2. `"MatConvNet - Convolutional Neural Networks for MATLAB", A. Vedaldi and K. Lenc, Proc. of the ACM Int. Conf. on Multimedia, 2015. `
3. `https://github.com/hbilen/mcnDCGAN`
