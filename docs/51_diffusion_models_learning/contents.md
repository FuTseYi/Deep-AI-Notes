# 第 51 期《扩散模型从原理到实战》学习笔记

本次任务需要提前配置代理

## 设置 Jupyter Notebook 代理

```shell
set HTTPS_PROXY=http://127.0.0.1:19180
set HTTP_PROXY=http://127.0.0.1:19180
```

设置代理之后，启动 Jupyter Notebook

```shell
jupyter notebook
```

## Task01 扩散模型库入门与零基础实战

本次任务，主要了解了扩散模型的基本原理（前向过程、反向过程、优化目标），并学习了 Hugging Face 核心功能的使用，根据教程从零搭建了扩散模型，并将 BasicUNet 和 UNet2Model 两个模型预测结果进行对比。

个人笔记如下：

- [第1章 扩散模型简介](51_diffusion_models_learning/ch01.md)
- [第2章 Hugging Face简介](51_diffusion_models_learning/ch02.md)
- [第3章 从零开始搭建扩散模型](51_diffusion_models_learning/ch03/ch03.md)

## Task02 微调与引导

本次任务，主要基于 Diffusers 库，了解 Diffusers 核心 API（管线、模型和调度器），通过蝴蝶图像生成的代码实战，学习扩散模型从定义、训练到图像生成一系列的实战内容；并了解微调和引导这两个技术，通过 CLIP 引导可以用文字描述控制一个没有条件约束的扩散模型的生成过程。

个人笔记如下：

- [第4章 Diffusers实战](51_diffusion_models_learning/ch04/ch04.md)
- [第5章 微调和引导](51_diffusion_models_learning/ch05/ch05.md)

## Task03 Stable Diffusion 原理与实战

本次任务，主要介绍了 Stable Diffusion 文本条件隐式扩散模型，包括隐式扩散、文本条件生成的概念，结合 Stable Diffusion 默认使用的管线，使用提示语句生成图片，还介绍了其他管线，比如 Img2Img、Depth2Img，这两个管线的生成效果也非常优秀。

个人笔记如下：

- [第6章 Stable Diffusion](51_diffusion_models_learning/ch06/ch06.md)

## Task04 DDIM 反转与音频扩散模型

本次任务，主要介绍了 DDIM 反转中采样的工作原理，反转其实就是在时间步上朝着相反的方向移动，向噪声更多的方向移动，通过反转得到原始图像。使用 ControlNet 结合 prompt，可以生成很棒的图片。还介绍了音频扩散模型，使用 Audio Diffusion 结合 prompt 提示的风格，可以生成类似风格的频谱和音频。

个人笔记如下：

- [第7章 DDIM反转](51_diffusion_models_learning/ch07/ch07.md)
- [第8章 音频扩散模型](51_diffusion_models_learning/ch08/ch08.md)
