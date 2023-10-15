---
layout: post
title: 论文精读-GloVe—Global Vectors for Word Representation
date: 2023-10-14
Author: YMieMie
tags: [nlp]
toc: true
comments: true
---

读经典的nlp论文：GloVe: Global Vectors for Word Representation

# Abstract

结果是一个新的 **global log-bilinear regression model**，它结合了文献中两大模型族的优点:**global matrix factorization** 和 **local context window methods**。

我们的模型通过只训练a word-word cooccurrence matrix中的非零元素，而不是整个稀疏矩阵或大型语料库中的单个上下文窗口，有效地利用了统计信息。

# 1 Introduction

目前，这两个**model famlies**都有明显的缺陷。虽然像**latent semantic analysis (LSA)**这样的方法有效地利用了统计信息，但它们在单词类比任务上做得相对较差，这表明向量空间结构不是最优的。像**skip-gram**这样的方法可能在类比任务上做得更好，但它们很少利用语料库的统计数据，因为它们在单独的局部上下文窗口上训练，而不是在全局共出现计数上训练。

[模型源代码](http://nlp.stanford.edu/projects/glove/)。

# 2 Related Work

## Matrix Factorization Methods

**Hyperspace Analogue to Language(HAL)** ，例如，使用“term-term”类型的矩阵，即行和列对应于单词，条目对应于给定单词在另一个给定单词的上下文中出现的次数。

出现问题，两个单词与or and同时出现的次数会对它们的相似性产生很大的影响，尽管它们的语义相关性传达的相对较少，使用一些归一化和各种压缩方法解决该问题。

## Shallow Window-Based Methods

另一种方法是学习单词表示，其有助于在局部下文窗口内进行预测。

在skip-gram和ivLBL模型中，目标是在给定单词本身的情况下预测单词的上下文，而在CBOW和vLBL模型中，目标是在给定上下文的情况下预测单词。

这些模型扫描整个语料库的上下文窗口，这无法利用大量重复的数据。

# 3 The GloVe Model

我们利用我们的见解构建了一个新的词表示模型，我们称之为GloVe，即global vector，因为全局语料库统计数据是由模型直接捕获的。

[![pi9pYTO.png](https://z1.ax1x.com/2023/10/14/pi9pYTO.png)](https://imgse.com/i/pi9pYTO)

上述论点表明，单词向量学习的适当起点应该是ratios of co-occurrence probabilities，而不是the probabilities themselves。

[![pi9p161.png](https://z1.ax1x.com/2023/10/14/pi9p161.png)](https://imgse.com/i/pi9p161)

$w$ : word vectors

$\widetilde w $ : separate context word vectors

最后一步步优化的函数：

[![pi9pGm6.png](https://z1.ax1x.com/2023/10/14/pi9pGm6.png)](https://imgse.com/i/pi9pGm6)

该模型的一个主要缺点是，它对所有共同发生的情况一视同仁，即使是那些很少发生或从未发生的情况。这种罕见的共存是嘈杂的，比更频繁的共存携带的信息更少——然而，根据词汇量和语料库的大小，即使是零条目也占X中数据的75-95%。

我们使用了least squares regression model来解决这个问题，而且引入了一个权重函数$f(X_{ij})$:

[![pi9pJ0K.png](https://z1.ax1x.com/2023/10/14/pi9pJ0K.png)](https://imgse.com/i/pi9pJ0K)

其中V是词汇量的大小。挑选出的效果好的一个优化函数：

[![pi9pllR.png](https://z1.ax1x.com/2023/10/14/pi9pllR.png)](https://imgse.com/i/pi9pllR)

## 3.1 Relationship to Other Models

在本小节中，我们将展示其他的基于语料库的模型与我们定义的模型的关联。

[![pi9pJ0K.png](https://z1.ax1x.com/2023/10/14/pi9pJ0K.png)](https://imgse.com/i/pi9pJ0K)



## 3.2 Complexity of the model

权重函数 $f(x)$ 的形式，很容易发现，模型的复杂度的计算是依赖于词共现矩阵 $X$中的非0元素的；自然复杂度是不会超过 $O(|V|^2$) ，咋一看，这对于浅层窗口的方法(如word2vec)来说，应该算是一个提升；毕竟计算规模缩减成了 $|V|$ ；但是对于实际情况，词汇表的大小是成千上万的，以至于 $|V|^2$ 的大小可能是上亿的，这实际比大多数语料库大得多。鉴于这个原因，是否可以对 $X$ 的非零元素的数目施加一个更紧的界就变得很重要了。

# 4 Experiments

## 4.1 Evaluation methods

### Word analogies  

任务类似 *“a is to b as c is to ?”*

a semantic subset and a syntactic subset 

### Word similarity

虽然类比任务是我们的主要关注点，因为它测试了有趣的向量空间子结构，但我们也在表3中的各种单词相似度任务上评估了我们的模型。

​                                                      **Spearman在单词相似度任务上对相关性进行排序**

![pi9P01J.png](https://z1.ax1x.com/2023/10/14/pi9P01J.png)





### Named entity recognition

NER的CoNLL-2003英语基准数据集是来自路透社新闻通讯社文章的文档集合，用四种实体类型进行注释:人、地点、组织和杂项。

## 4.2 Corpora and training details

我们在五个不同大小的语料库上训练我们的模型。

在所有情况下，我们都使用一个递减的权重函数，这样相隔d个单词的单词对对总数的贡献为1/d。

参数的设置$x_{max} = 100$ 和 $ \alpha = 3/4 $

除非另有说明，否则我们使用左边10个单词，右边10个单词的上下文。

该模型生成两组词向量，$W$和$\widetilde W$。当$X$是对称的，$W$和$\widetilde W$是等价的，只是由于它们的随机初始化而不同;这两组向量应该是相等的，使用$W + \widetilde W$作为word vetors，在语义类比任务上会有很大的提升。

## 4.3 Results

​                                                                             **word analogy task的结果**

[![pi9pUte.png](https://z1.ax1x.com/2023/10/14/pi9pUte.png)](https://imgse.com/i/pi9pUte)

​                                                                            **基于crf的模型在NER任务上的结果**

[![pi9pNkD.png](https://z1.ax1x.com/2023/10/14/pi9pNkD.png)](https://imgse.com/i/pi9pNkD)

## 4.4 Model Analysis: Vector Length and Context Size

​                                                       **不同向量长度和上下文窗口的实验结果**

[![pi9PBc9.png](https://z1.ax1x.com/2023/10/14/pi9PBc9.png)](https://imgse.com/i/pi9PBc9)

扩展到目标单词左右的上下文窗口称为对称窗口，而仅向左扩展的上下文窗口称为非对称窗口。

## 4.5 Model Analysis: Corpus Size

​                                                        **在不同语料库上训练的300维向量类比任务的准确性**

[![pi9PahF.png](https://z1.ax1x.com/2023/10/14/pi9PahF.png)](https://imgse.com/i/pi9PahF)

## 4.6 Model Analysis: Run-time

总运行时间分为填充X和训练模型。

前者取决于许多因素，包括窗口大小、词汇量大小和语料库大小。

给定X，训练模型所需的时间取决于向量大小和迭代次数。                                                                                  

## 4.7 模型分析:与word2vec的比较

由于存在许多对性能有强烈影响的参数，因此对GloVe和word2vec进行严格的定量比较变得复杂。

需要控制的最重要的变量是训练时间。

对于GloVe，相关参数是训练迭代的次数。

对于word2vec，最明显的选择是训练周期的数量。

[![pi9Pwp4.png](https://z1.ax1x.com/2023/10/14/pi9Pwp4.png)](https://imgse.com/i/pi9Pwp4)

我们注意到，如果负样本的数量增加到大约10个以上，word2vec的性能实际上会下降。

# 5 Conclusion

我们构建了一个模型，利用计数数据的这一主要优势，同时捕获在最近基于对数双线性预测的方法(如word2vec)中普遍存在的有意义的线性子结构。结果，GloVe是一个新的全局对数双线性回归模型，用于单词表示的无监督学习，在单词类比、单词相似度和命名实体识别任务上优于其他模型。

