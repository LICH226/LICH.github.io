---
layout: post
title: 论文精读-Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
date: 2023-10-17
Author: YMieMie
tags: [nlp]
toc: true
comments: true
---
nlp打基础读论文Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention。

# Abstract

Transformer 在几个任务中取得了显着的性能，但由于它们的二次复杂度，相对于输入的长度，对于非常长的序列来说它们太慢了。为了解决这个限制，我们将self-attention表示为a linear dot-product of kernel feature maps，并利用矩阵乘积的结合性属性将复杂度从$\Omicron(N^2)$降低到 $\Omicron(N)$，其中 $N$ 是序列长度。我们这个公式允许迭代，可以显著实现加速自回归transformer，并揭示了它与$RNN$的关系。我们的*linnear transformer*实现了与*vanilla transformer*相似的性能，并且在非常长的序列的自回归预测上快高达4000倍。

# 1 Introduction

在本文中，我们介绍了*linear Transformer* 模型，该模型显着减少了内存占用并与上下文长度成线性关系。我们通过使用kernal-based的自注意力公式和the associative property of matrix products 来计算自注意力权重来实现这一点（第 3.2 节）。使用我们的线性公式，我们还用linear complexity and constant memory表达the associative property of matrix products t（第 3.3 节）。这揭示了转换器和 RNN 之间的关系，这使我们能够更快地执行自回归推理（第 3.4 节）。

# 2 Related Work

## 2.1 Efficient Transformers



# Conclusions

在这项工作中，我们提出了*linnear transformer*，这是一种显着降低*vanilla transformer*内存和计算成本的模型。特别是，通过利用矩阵乘积的结合性属性，我们能够以相对于序列长度呈线性扩展的时间和空间计算自注意力以。我们表明，我们的模型可以用于*casual masking*，并且仍然保持其线性复杂度。最后，我们将 Transformer 模型表示为循环神经网络，这使我们能够以数千时间的速度对自回归任务进行推理。
