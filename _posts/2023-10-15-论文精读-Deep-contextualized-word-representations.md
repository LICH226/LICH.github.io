---
layout: post
title: 论文精读-Deep contextualized word representations
date: 2023-10-15
Author: YMieMie
tags: [nlp]
toc: true
comments: true
---
精度论文Deep contextualized word representations

# Abstract

我们引入了一种新型的 **deep contextualized**的词表示方法，它不仅可以模拟词使用的复杂特征（包括语义和句法），以及这些词在不同语境下的不同用法（model polysemy）。我们的词向量表示从深度双向语言模型（一个在大量文本语料库的预训练的deep bidirections **biLM**）的内部状态学习。我们发现这些模型表示可以轻松加入到这些模型中，并且极大的提高六个nlp任务的效果。**我们发现研究预训练网络的深层内部是至关重要的，它利于下游模型混合不同类型的半监督信号。**

# 1 Introduction

通过使用内在评估，我们发现更高级别的LSTM状态捕获了单词含义的上下文相关方面(例如，它们可以在不修改的情况下用于执行有监督的词义消歧任务)，而更低级别的状态则建模了语法方面(例如，它们可以用于词性标注)。

[模型代码](http://allennlp.org/elmo
)

# 2 Related work

在本文中，我们充分利用了对大量单语数据的访问，并在大约3000万个句子的语料库上训练我们的biLM。我们还将这些方法推广到深度上下文表示，证明这些方法在各种各样的NLP任务中都能很好地工作。

先前的研究表明，不同层次的深度 biRNNs 编码不同类型的信息。例如，在 deep LSTM的较低层次引入**多任务语法监督**(例如，词性标签)可以提高更高层次任务的整体性能，如依赖解析等。

在基于RNN的编码器-解码器机器翻译系统中，Belinkov等人发现，在2层LSTM编码器中，第一层学习的表征比第二层更能预测POS标签，而用于编码单词上下文的LSTM的顶层被证明可以学习词义的表示。

# 3 ELMo: Embeddings from Language Models

![image-20231015150409448](C:\Users\15295\AppData\Roaming\Typora\typora-user-images\image-20231015150409448.png)

## 3.1 Bidirectional language models

a forward language model

[![pi9fUOI.png](https://z1.ax1x.com/2023/10/15/pi9fUOI.png)](https://imgse.com/i/pi9fUOI)

a backward language model

[![pi9fYSH.png](https://z1.ax1x.com/2023/10/15/pi9fYSH.png)](https://imgse.com/i/pi9fYSH)

我们的公式共同最大化了正向和反向的对数似然:

[![pi9fN6A.png](https://z1.ax1x.com/2023/10/15/pi9fN6A.png)](https://imgse.com/i/pi9fN6A)

我们将令牌表示($\Theta_x$)和Softmax层($\Theta_s$)的参数在正向和反向上绑定，同时在每个方向上为LSTM保留单独的参数。

## 3.2 ELMo

对于每个token $t_K$ ，一个L层的biLM模型计算出 $ 2L+1$个词向量表示。

[![pi9ftld.png](https://z1.ax1x.com/2023/10/15/pi9ftld.png)](https://imgse.com/i/pi9ftld)

一个token layer层的，每个biLSTM层，都有两个方向的向量表示。

为了包含在下游模型中，ELMo将$R$中的所有层折叠成单个向量，$ELMo_k = E(R_k;\Theta_e)$。

一般情况下，我们计算所有biLM层的任务特定权重:

[![pi9fdmt.png](https://z1.ax1x.com/2023/10/15/pi9fdmt.png)](https://imgse.com/i/pi9fdmt)

$s_{task}$是softmax归一化的权重，标量参数$\gamma_{task}$允许任务模型缩放整个ELMo向量。$\gamma$对于帮助优化过程具有实际重要性。

## 3.3 Using biLMs for supervised NLP tasks

我们首先运行biLM，记录录每个单词的所有层的表示。然后让这个任务模型学习线性组合这些向量表示。

首先我们考虑没有biLM的监督架构的低层。这些监督架构的低层大多数共享一个相同的架构，方便我们统一加入ELMo。

给定一个令牌序列$(t_1,...,t_N)$，使用预训练的词嵌入和可选的基于字符的表示，为每个标记位置形成与上下文无关的标记表示$x_k$。然后，该模型形成一个上下文敏感的表示$h_k$，通常使用双向$RNN$、$CNN$或feed forward networks。

最后，我们发现在ELMo中添加适量的dropout，或在某些情况下，通过向损失中添加$\lambda \parallel x \parallel _2^2$来正则化ELMo权重都是有益的。这对ELMo权重施加了一个电感偏置，以保持接近所有biLM层的平均值。

## 3.4 Pre-trained bidirectional language model architecture

本文中预训练的biLM类似于Jozefowicz等人(2016)和Kim等人(2015)的架构，但进行了修改，以支持两个方向的联合训练，并在LSTM层之间添加了残差连接。

最终的模型映射层（单词到word embedding）使用的Jozefowicz的CNN-BIG-LSTM，即输入为512维的列向量。

一共使用L = 2个biLSTM层，4096个单元和512个维度的投影，以及从第一层到第二层的残差连接。

经过预训练后，biLM可以计算任何任务的表示。在某些情况下，对特定领域数据的biLM进行微调可以显著降低困惑度并提高下游任务性能。这可以看作是biLM的一种领域转移。因此，在大多数情况下，我们在下游任务中使用了经过微调的biLM。

# 4 Evaluation

​                                        **在六个基准NLP任务中，ELMo增强神经模型与最先进的单一模型基线的测试集比较**

[![pi9fDk8.png](https://z1.ax1x.com/2023/10/15/pi9fDk8.png)](https://imgse.com/i/pi9fDk8)

# 5 Analysis

## 5.1 Alternate layer weighting schemes

正则化参数$\lambda$的选择也很重要，因为较大的值(例如$\lambda$= 1)有效地将权重函数减少到层间的平均值，而较小的值(例如$\lambda$ = 0:001)允许层权重变化。

​                             **在开发集上，使用baseline、只使用最上层，和不同的$\lambda$的三个任务效果**

[![pi9fw0P.png](https://z1.ax1x.com/2023/10/15/pi9fw0P.png)](https://imgse.com/i/pi9fw0P)

## 5.2 Where to include ELMo?

​                                   **当在监督模型中包含ELMo时，开发集中SQuAD, SNLI和SRL的效果**

[![pi9f0Tf.png](https://z1.ax1x.com/2023/10/15/pi9f0Tf.png)](https://imgse.com/i/pi9f0Tf)

## 5.3 What information is captured by the biLM’s representations?

[![pi9HzSf.png](https://z1.ax1x.com/2023/10/15/pi9HzSf.png)](https://imgse.com/i/pi9HzSf)

底部两行显示了SemCor数据集(见下文)中使用源句子中“play”的biLM上下文表示的最近邻句子。在这些情况下，biLM能够消除源句子中的词性和词义的歧义。

为了隔离由biLM编码的信息，这些表示用于直接预测细粒度的词义消歧(WSD)任务和POS标记任务。

​                     **所有字细粒度WSD的F1分数。对于CoVe和biLM，我们记录了第一层和第二层biLSTMs的分数**

[![pi9HjYt.png](https://z1.ax1x.com/2023/10/15/pi9HjYt.png)](https://imgse.com/i/pi9HjYt)

​                    **测试集PTB的POS标记准确性。对于CoVe和biLM，我们记录了第一层和第二层biLSTMs的分数**

[![pi9HXFI.png](https://z1.ax1x.com/2023/10/15/pi9HXFI.png)](https://imgse.com/i/pi9HXFI)

**Implications for supervised tasks**

综上所述，这些实验证实了biLM中的不同层代表不同类型的信息，并解释了为什么包括所有biLM层对于下游任务的最高性能很重要。此外，biLM的表示比CoVe中的表示更易于转移到WSD和POS标记中，这有助于说明为什么ELMo在下游任务中优于CoVe。

## 5.4 Sample efficiency

论是在达到最先进性能的参数更新数量方面，还是在总体训练集大小方面，将ELMo添加到模型中都大大提高了样本效率。

此外，ELMo增强模型比没有ELMo的模型更有效地使用更小的训练集。

​                               **当训练集大小从0.1%到100%变化时，SNLI和SRL的基线性能与ELMo性能的比较**

[![pi9HLTA.png](https://z1.ax1x.com/2023/10/15/pi9HLTA.png)](https://imgse.com/i/pi9HLTA)

## 5.5 Visualization of learned weights

​                                                              **图2显示了softmax归一化的学习层权重**

[![pi9Hqwd.png](https://z1.ax1x.com/2023/10/15/pi9Hqwd.png)](https://imgse.com/i/pi9Hqwd)

在输入层，任务模型倾向于第一个biLSTM层。对于coreference和SQuAD，这是非常受欢迎的，但其他任务的分布没有达到峰值。输出层的权重相对平衡，对较低的层有轻微的偏好。

# 6 Conclusion

我们介绍了一个通用的方法去学习高质量的深层次的上下文相关的representations，展示了巨大的性能提升在运用ELMo在广泛的NLP任务上。通过消融和其它的控制实验，我们证实biLM层有效地编码了上下文中不同类型的句法和语义信息，并且结合所有层可以提高整体任务的性能。









