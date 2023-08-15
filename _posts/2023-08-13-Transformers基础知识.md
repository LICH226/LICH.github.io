---
layout: post
title: Transformers基础知识
date: 2023-08-13
Author: YMieMie
tags: [ NLP]
toc: true
comments: true
---

记录一下transformers库的学习

# Transformers基础知识及相关库

●Transformers: 核心库，模型加载、模型训练、流水线等
●Tokenizer: 分词器，对数据进行预处理，文本到token序列的互相转换
●Datasets: 数据集库,提供了数据集的加载、处理等方法
●Evaluate: 评估函数,提供各种评价指标的计算函数
●PEFT:高效微调模型的库，提供了几种高效微调的方法，小参数量撬动大模型
●Accelerate: 分布式训练，提供了分布式训练解决方案，包括大模型的加载与推理解决方案
●Optimum: 优化加速库，支持多种后端， 如Onnxruntime、 OpenVino等
●Gradio: 可视化部署库，几行代码快速实现基于Web交互的算法演示系统

# 基础组件之Pipline

![image-20230814180034561](C:\Users\15295\AppData\Roaming\Typora\typora-user-images\image-20230814180034561.png)

# 基础组件之tokenizer
