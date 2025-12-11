# Task07 PyTorch 生态简介

## 1 torchvision（图像）

- torchvision.datasets：计算机视觉领域常见的数据集，包括 CIFAR、EMNIST、Fashion-MNIST 等
- torchvision.transforms：数据预处理方法，可以进行图片数据的放大、缩小、水平或垂直翻转等
- torchvision.models：预训练模型，包括图像分类、语义分割、物体检测、实例分割、人体关键点检测、视频分类等模型
- torchvision.io：视频、图片和文件的 IO 操作，包括读取、写入、编解码处理等
- torchvision.ops：计算机视觉的特定操作，包括但不仅限于 NMS，RoIAlign（MASK R-CNN 中应用的一种方法），RoIPool（Fast R-CNN 中用到的一种方法）
- torchvision.utils：图片拼接、可视化检测和分割等操作

## 2 PyTorchVideo（视频）

- 简介：PyTorchVideo 是一个专注于视频理解工作的深度学习库，提供加速视频理解研究所需的可重用、模块化和高效的组件，使用 PyTorch 开发，支持不同的深度学习视频组件，如视频模型、视频数据集和视频特定转换。
- 特点：基于 PyTorch，提供 Model Zoo，支持数据预处理和常见数据，采用模块化设计，支持多模态，优化移动端部署
- 使用方式：[TochHub](https://pytorchvideo.org/docs/tutorial_torchhub_inference)、[PySlowFast](https://github.com/facebookresearch/SlowFast/)、[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

## 3 torchtext（文本）

- 简介：torchtext 是 PyTorch 的自然语言处理（NLP）的工具包，可对文本进行预处理，例如截断补长、构建词表等操作
- 构建数据集：使用 `Field` 类定义不同类型的数据
- 评测指标：使用 `torchtext.data.metrics` 下的方法，对 NLP 任务进行评测

## 4 总结

&emsp;&emsp; 本次任务，主要介绍了 PyTorch 生态在图像、视频、文本等领域中的发展，并介绍了相关工具包的使用。

1. 图像：torchvision 主要提供在计算机视觉中常常用到的数据集、模型和图像处理操作。
2. 视频：PyTorchVideo 主要基于 PyTorch，提供 Model Zoo，支持数据预处理和常见数据，采用模块化设计，支持多模态，优化移动端部署。
3. 文本：torchtext 是 PyTorch 的自然语言处理（NLP）的工具包，可对文本进行预处理，例如截断补长、构建词表等操作。
