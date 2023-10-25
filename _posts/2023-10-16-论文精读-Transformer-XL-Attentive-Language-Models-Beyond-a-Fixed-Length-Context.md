---
layout: post
title: 论文精读-Transformer-XL
date: 2023-10-16
Author: YMieMie
tags: [nlp]
toc: true
comments: true
---

nlp打基础，读论文Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

# Abstract

Transformers 具有学习长期依赖关系的潜力，但在语言建模设置中受到固定长度上下文的限制。我们提出了一种新的神经架构 **Transformer-XL**，它能够在不破坏时间一致性的情况下学习固定长度之外的依赖关系。它由一个**segment-level recurrence mechanism**和一个新颖的**positional encoding scheme**组成。我们的方法不仅可以**捕获长期依赖关系**，还可以解决**上下文碎片问题**。因此，TransformerXL 可以学习比 RNN 长 80% 和比普通 Transformer 长 450% 的依赖关系，在短序列和长序列上都取得了更好的性能，在评估过程中比普通 Transformer 快 1,800+ 倍。值得注意的是，我们在 enwiki8 上将 bpc/perplexity 的最新结果提高到 0.99，在 text8 上提高了 1.08，在 WikiText-103 上提高了 18.3，在十亿字上提高了 21.8，在 Penn Treebank 上提高了 54.5（没有微调）。仅在 WikiText-103 上训练时，Transformer-XL 可以生成具有数千个token的合理连贯、新颖的文本文章。我们的[代码](https://github.com/kimiyoung/transformer-xl)、预训练模型和超参数都可以在 Tensorflow 和 PyTorch 中找到。

# 1 Introduction

尽管取得了成功，al - rfu等人(2018)的LM训练是在几百个字符的固定长度的独立片段上进行的，没有任何跨片段的信息流。固定上下文长度的结果是，模型无法捕获超出预定义上下文长度的任何长期依赖关系。此外，通过选择连续的符号块来创建固定长度的片段，而不考虑句子或任何其他语义边界。因此，该模型缺乏必要的上下文信息来很好地预测前几个符号，导致低效的优化和较差的性能。我们把这个问题称为**context fragmentation**。

为了解决前面提到的固定长度上下文的限制，我们提出了一个名为**Transformer-XL**(意思是超长)的新体系结构。

特别是，我们不再为每个新段从头计算隐藏状态，而是重用在以前的段中获得的隐藏状态。

重用的隐藏状态充当当前段的内存，从而在段之间建立循环连接。因此，建模非常长期的依赖关系成为可能，因为信息可以通过循环连接传播。更重要的是，我们展示了使用**相对位置编码**（比普通注意力机制可以注意到更长的文本）而不是绝对位置编码的必要性，以便在不造成时间混乱的情况下实现状态重用。

Transformer-XL是第一个在字符级和单词级语言建模上都比$RNN$取得更好结果的自注意模型。

# 2 Related Work

更广泛地说，在通用序列建模中，如何捕获长期依赖关系一直是一个长期存在的研究问题。

我们的工作是基于Transformer体系结构的，并表明语言建模作为一项现实世界的任务，可以从学习长期依赖关系的能力中获益。

# 3 Model

## 3.1 Vanilla Transformer Language Models

核心问题是如何应用transformer和self-attention来解决长文本。

一种可实现但是粗略的方法是将长文本分割成短的分段，只在每个小段上训练模型，而忽略前面段的上下文信息。我们称之为**vanilla model**。然而，简单地将序列分成固定长度的片段将导致上下文碎片问题

<div align = center><img src = "https://z1.ax1x.com/2023/10/17/piC7I0J.png"</div>

在评估过程中，每一步**vanilla model**也消耗与训练中相同长度的段，但只在最后一个位置进行一次预测。然后，在下一步中，段只向右移动一个位置，并且必须从头开始处理新段。如图所示，该过程确保了每个预测都利用了训练过程中暴露的最长的上下文，也缓解了训练中遇到的上下文碎片化问题。然而，这个评估过程非常昂贵。我们将展示我们提出的体系结构能够大大提高评估速度。

<div align = center><img src = "https://z1.ax1x.com/2023/10/17/piC7bfx.png"</div>

## 3.2 Segment-Level Recurrence with State Reuse

为了解决使用固定长度上下文的限制，我们建议在Transformer体系结构中引入一种循环机制。在训练过程中，为前一个段计算的隐藏状态序列被固定并缓存，以便在模型处理下一个新段时作为扩展上下文重用，如图2a所示。

<div align = center><img src = "https://z1.ax1x.com/2023/10/17/piC7O1K.png"</div>

<div align = center><img src = "https://z1.ax1x.com/2023/10/17/piC75m4.png"</div>

$$
h_{\tau}^n是第\tau个序列段的第n层的隐藏状态\mathbb{R}^{L\times d}
\newline
SG()代表梯度停止
\newline
[h_u,h_v]表示隐藏状态的concatenation
\newline
\widetilde h_{\tau+1}^{n-1}拓展的上下文向量
$$
因此，最大可能的依赖长度随层数和段长度线性增长，即O(N × L)，如图2b中的阴影区域所示。

<div align = center><img src = "https://z1.ax1x.com/2023/10/17/piC7bfx.png"</div>



除了实现超长上下文和解决碎片之外，递归方案带来的另一个好处是显著加快了计算速度。具体来说，在评估过程中，可以重用来自前面部分的表示，而不是像普通模型那样从头开始计算。

我们的实验中，我们在训练时将M设置为片段长度，在评估时将其增加数倍。

## 3.3 Relative Positional Encodings

出于同样的目的，人们可以将相同的信息注入到每一层的注意力分数中，而不是静态地将偏差合并到初始嵌入中($E_{s_{\tau}} \in \mathbb{R}^{L \times d}$是$s_{\tau}$段的word embeding)。
$$
E_{s_{\tau}} \in \mathbb{R}^{L \times d}是s_{\tau}段的word-embeding
\newline
U \in \mathbb{R}^{L_{max} \times d}中的第i行U_i是第i的绝对位置编码
$$
举个例子，当一个query vector $q_{\tau,i}$去attend这些key vectors $k_{\tau,\leqslant i}$，它不需要知道每个key vector的绝对位置去识别段的时间顺序。相反的，知道key  vector $k_{\tau,j}$和它本身query vector $q_{\tau,i}$之间的相对距离就足够了。因此创造一个相对位置编码R $\in$$ \mathbb{R}^{L_{max} \times d}$，第$i$行$R_i$表示相对位置为$i$的位置向量。

同时，我们不会丢失任何时间信息，因为绝对位置可以从相对距离递归恢复。

首先，在标准 Transformer (Vaswani et al., 2017) 中，同一段内query vector $q_i$ 和key vector $k_j$ 之间的注意力分数可以分解为

<div align = center><img src = "https://z1.ax1x.com/2023/10/17/piC77kR.png"</div>

其中，$E_{x_i}$ 是词$i$的embedding，$E_{x_j}$是词$j$的embeddn是位置向量，这个式子实际上是$(W_q(E_{x_i}+U_i))^T \cdot (W_k(E_{x_j}+U_j))$的展开，就是Transformer中的标准格式。

遵循仅依靠相对位置信息的思想，我们建议重新参数化四个术语如下:

<div align = center><img src = "https://z1.ax1x.com/2023/10/17/piC7Ht1.png"</div>

对比来看，主要有三点变化：

- 在(b)和(d)这两项中，将所有绝对位置向量$U_j$都转为相对位置向量$R_{i-j}$，与Transformer一样，这是一个固定的编码向量，不需要学习。
- 在(c)这一项中，将查询的$U_i^T W_q^T$向量转为一个需要学习的参数向量$u$，因为在考虑相对位置的时候，不需要查询的绝对位置$i$，因此对于任意的$i$，都可以采用同样的向量。同理，在(d)这一项中，也将查询的$U_i^T W_q^T$向量转为另一个需要学习的参数向量$v$。
- 将键的权重变换矩阵$W_k$转为$W_{k,E}$和$W_{k,R}$，分别作为content-based key vectors和location-based key vectors。

从另一个角度来解读这个公式的话，可以将attention的计算分为如下四个部分：

- 基于内容的“寻址”，即没有添加原始位置编码的原始分数。
- 基于内容的位置偏置，即相对于当前内容的位置偏差。
- 全局的内容偏置，用于衡量key的重要性。
-  全局的位置偏置，根据query和key之间的距离调整重要性。



# 5 Conclusions

Transformer-XL 获得了强大的困惑度结果，比 RNN 和 Transformer 建模长期依赖关系，在评估期间实现了显着的加速，并且能够生成连贯的文本文章。我们设想 Transformer-XL 在文本生成、无监督特征学习、图像和语音建模领域的有趣应用。
