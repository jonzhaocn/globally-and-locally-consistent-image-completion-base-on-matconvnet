# global-and-local-consistent-image-completion-base-on-matconvnet

## 说明
使用`matconvnet`实现`Globally and locally consistent image completion`

1. 运行前需要先安装`matconvnet`，`matconvnet`主页在[这里](http://www.vlfeat.org/matconvnet/)（本项目在`matconvnet 1.0-beta25`上搭建），`matconvnet`安装在`matlab`的`toolbox`文件夹下；

2. 安装好`matconvnet`之后，将本项目中的文件复制到`matconvnet`的`examples`文件夹下（可以先在`example`文件夹中创建一个新文件夹，再将文件复制到其中）

3. 项目文件的训练的入口函数是`glcic_train()`，直接运行`glcic_train.m`开始训练；

4. 在训练过程中会保存当前训练的部分修复结果，即网络对训练数据的修复结果，作为参考使用，这部分的代码在`process_epoch`函数中

## 注意事项
#### 使用多gpu训练网络
* 先要运行`vl_setupnn.m`，也就是把下面这些路径添加到matlab path中，然后重启matlab
```
root = vl_rootnn() ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'matlab', 'mex')) ;
addpath(fullfile(root, 'matlab', 'simplenn')) ;
addpath(fullfile(root, 'matlab', 'xtest')) ;
addpath(fullfile(root, 'examples')) ;
```
* 不要将`compatibility`这个文件夹的内容添加到`path`中，文件夹中的`labindex.m`和`numlabs.m`会使程序无法正常获取到多个`core`
* 如果运行程序的时候出现下面类似的错误，可以将涉及到的函数移动到一个临时文件夹中。
```
Attempt to execute SCRIPT vl_imreadjpeg as a function:
    /home/liyanran/Desktop/MATLAB/R2017a/toolbox/matconvnet-1.0-beta25/matlab/vl_imreadjpeg.m
```
这些影响到程序运行的文件都只是脚本，其完整的功能在`mex`文件夹对应的程序文件中，所以移动这些`.m`文件不会影响到程序的运行。移动之后再次将`matlab`文件夹进行`addpath`，再重启`matlab`。

```
root = vl_rootnn() ;
addpath(fullfile(root, 'matlab')) ;
```



## 参考文献

1. `Ishikawa H, Ishikawa H, Ishikawa H. Globally and locally consistent image completion[M]. ACM, 2017.`
2. `"MatConvNet - Convolutional Neural Networks for MATLAB", A. Vedaldi and K. Lenc, Proc. of the ACM Int. Conf. on Multimedia, 2015. `
3. `https://github.com/hbilen/mcnDCGAN`
