---
layout: post
title: 论文精读-Neural Machine Translation by Jointly Learing to Align and Translate
date: 2023-10-15
Author: YMieMie
tags: [nlp]
toc: true
comments: true
---
nlp打基础，阅读论文Neural Machine Translation by Jointly Learing to Align and Translate。

# Abstract

神经机器翻译是最近提出的一种机器翻译方法。与传统的统计机器翻译不同，神经机器翻译旨在构建单个神经网络，这些神经网络可以共同调优，以最大限度地提高翻译性能。最近提出的神经机器翻译模型通常属于编码器-解码器家族，它们将源句子编码为固定长度的向量，解码器从中生成翻译。在本文中，我们推测使用固定长度向量是提高基本编码器-解码器架构性能的瓶颈，**并提议通过允许模型自动(soft-)搜索源句子中与预测目标词相关的部分，而不必将这些部分明确地形成硬段来扩展这一点**。通过这种新方法，我们在英法翻译任务上实现了与现有最先进的基于短语的系统相当的翻译性能。此外，定性分析表明，模型发现的(soft-)alignments与我们的直觉很好地吻合。

# 1 INTRODUCTION

传统地编码器-解码器方法的一个潜在问题是，神经网络需要能够将源句子的所有必要信息压缩成固定长度的向量。这可能会使神经网络难以处理长句子，特别是那些比训练语料库中的句子更长的句子。Cho等人(2014b)表明，基本编码器-解码器的性能确实会随着输入句子长度的增加而迅速恶化。

为了解决这个问题，我们引入了一种扩展的编码器-解码器模型，该模型可以学习对齐和翻译。每当提出的模型在翻译中生成一个单词时，它(soft-)在源句子中搜索最相关信息所集中的一组位置。然后，该模型根据这些源句子词中的位置和之前生成的所有目标单词相关的上下文向量来预测目标单词。

# 2 BACKGROUND: NEURAL MACHINE TRANSLATION

## 2.1 RNN ENCODER–DECODER

在这里，我们简要描述了Cho等人(2014a)和Sutskever等人(2014)提出的底层框架，称为$RNN  Encoder-Decoder$，我们在此基础上构建了一个学习align和同时translate的新架构。

# 3 LEARNING TO ALIGN AND TRANSLATE

在本节中，我们提出了一种新的神经机器翻译架构。新架构由一个双向RNN作为一个编码器(第3.2节)和一个解码器组成，该解码器在解码翻译过程中模拟搜索源句子(第3.1节)。

## 3.1 DECODER: GENERAL DESCRIPTION

<center>模型图示：</center>
<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCihtK.png"></div>
对于这个新模型，我们定义条件概率：

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCifk6.png"></div>

$s_i$ 为$RNN$时刻的隐藏状态，其计算公式：

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCigmR.png"></div>
注意到，对于每个目标词$y_i$，都有一个独特的上下文向量$c_i$。

上下文向量$c_i$依赖于编码器将输入句子映射到的一系列annotations($h_1、···、h_{T_x}$)。每个annotations都包含关于整个输入序列的信息，重点关注输入序列第i个单词周围的部分。我们将在下一节中详细解释如何计算annotations。

$c_i$计算为这些annotations的权重和：

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCiRTx.png"></div>

每个annotation $h_j$的权重$\alpha_{ij}$计算公式为：

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCi201.png"></div>

$e_{ij}$计算公式：

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCi4fO.png"></div>

概率$α_{ij}$ 或者其相关energy $e_{ij}$ 反映了annotations $h_j $相对于先前隐藏状态 $s_{i-1}$ 在决定下一个状态 $s_i$ 和生成 $y_i$ 中的重要性。直观地说，这在解码器中实现了一种注意力机制。解码器决定源句的部分来关注。通过让解码器具有注意力机制，我们减轻了编码器必须将源句子中的所有信息编码为固定长度的向量的负担。通过这种新方法，信息可以分布在整个注释序列中，解码器可以相应地选择性地检索。

## 3.2 ENCODER: BIDIRECTIONAL RNN FOR ANNOTATING SEQUENCES

我们使用了双向$RNN$模型，希望$h_j$可以不仅总结前面的$x$还可以总结后面的$x$。

我们通过concatenate前向隐藏状态$\mathop{h_j} \limits ^ {\rightarrow}$和后向隐藏状态$\mathop{h_j} \limits ^ {\leftarrow}$来获得每个词xj的注释，即$h_j = [\mathop{h_j} \limits ^ {\rightarrow},\mathop{h_j} \limits ^ {\leftarrow}]^T$。这样，annotations $h_j$ 包含前面单词和后面的单词的摘要信息。由于 $RNN$ 倾向于更好地表示最近的输入，annotations $h_j$ 将集中在 $x_j$ 周围的单词上。解码器使用这一系列注释，alignment model稍后用于计算上下文向量。

# 4 EXPERIMENT SETTINGS

## 4.1 DATASET

我们没有使用上面提到的平行语料库以外的任何单语数据，尽管可以使用更大的单语语料库来预训练编码器。

[实验数据](http://www.statmt.org/wmt14/translation-task.html)

[实验实现](https://github.com/lisa-groundhog/GroundHog)

## 4.2 MODELS

我们训练了两种类型的模型。第一个是 RNN Encoder-Decoder (RNNencdec, Cho et al., 2014a)，另一个是提出的模型，我们称之为 RNNsearch。我们两次训练每个模型：首先长度为 30 个单词的句子（RNNencdec-30、RNNsearch-30），然后是长度高达 50 个单词的句子（RNNencdec-50、RNNsearch-50）。

RNNencdec 的编码器和解码器各有 1000 个隐藏单元。RNNsearch 的编码器由前向和后向循环神经网络 (RNN) 组成，每个神经网络有 1000 个隐藏单元，它的解码器有 1000 个隐藏单元。

# 5 RESULTS

## 5.1 QUANTITATIVE RESULTS

<center>在测试集上计算的训练模型的 BLEU 分数</center>

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCiIpD.png"></div>


第二列和第三列分别显示所有句子的分数，和在本身和参考翻译中没有任何**UNK**单词的句子上。RNNsearch-50*训练时间更长，直到开发集的性能停止提高。

从表中可以看出，在所有情况下，所提出的 RNNsearch 都优于传统的 RNNencdec。更重要的是，当仅考虑由已知单词组成的句子时，RNNsearch 的性能与传统的基于短语的翻译系统 (Moses) 的性能一样高。这是一个重大成就，考虑到 Moses 除了用于训练 RNNsearch 和 RNNencdec 的并行语料库之外，还使用单独的单语语料库。



<center>测试集生成的翻译相对于句子长度的 BLEU 分数</center>

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCio1e.png"></div>
我们看到随着句子长度的增加，RNNencdec 的性能急剧下降。另一方面，RNNsearch-30 和 RNNsearch-50 都对句子的长度更稳健。RNNsearch50，尤其是即使句子长度为 50 或更多的句子，性能也不会恶化。

## 5.2 QUALITATIVE ANALYSIS

### 5.2.1 ALIGNMENT

<center>RNNsearch-50找到的四个sample alignments</center>

<div align=center><img src="https://z1.ax1x.com/2023/10/16/piCExVs.png"></div>

每个图中的矩阵的每一行表示与annotations相关的权重。由此我们看到在生成目标词时，源句中的哪些位置被认为更重要。

**soft-alignment**优于**hard-alignment**能考虑上下文向量，它的另一个好处是它自然地处理不同长度的源短语和目标短语，而不需要将一些单词映射到([NULL])的反直觉方法。

### 5.2.2 LONG SENTENCES

从图 2 可以清楚地看出，所提出的模型 (RNNsearch) 在翻译长句子方面比传统模型 (RNNencdec) 好得多。这可能是因为 RNNsearch 不需要完美地将长句子编码为固定长度的向量，而只需要准确地编码围绕特定单词的输入句子的部分。

结合已经提出的定量结果，这些定性观察证实了我们的假设，即 RNNsearch 架构比标准 RNNencdec 模型实现了更可靠的长句子翻译。

# 6 RELATED WORK

## 6.1 LEARNING TO ALIGN

我们的方法需要计算翻译中每个单词的源句子中每个单词的注释权重。这个缺点在翻译任务中并不严重，其中大多数输入和输出句子只有 15-40 个单词。然而，这可能会限制所提出方案对其他任务的适用性。

## 6.2 NEURAL NETWORKS FOR MACHINE TRANSLATION

我们更感兴趣的是设计一个基于神经网络的全新的翻译系统。因此，我们在本文中考虑的神经机器翻译方法与这些早期工作有着根本的不同。我们的模型不是使用神经网络作为现有系统的一部分，而是自己工作，直接从源语句生成翻译。

# 7 CONCLUSION

在本文中，我们通过在生成每个目标词时让模型（软）搜索一组输入词或编码器计算的annotations来扩展基本的编码器-解码器。这使模型能够将整个源句子编码为固定长度的向量，并让模型只关注与下一个目标词的生成相关的信息。这对神经机器翻译系统在较长句子上产生良好结果的能力有重大影响。与传统的机器翻译系统不同，所有翻译系统（包括alignment mechanism）都经过联合训练，以更好地生成正确翻译的对数概率。

未来留下的挑战之一是更好地处理未知或稀有词。这将要求模型更广泛地使用并匹配当前最先进的机器翻译系统在所有上下文中的性能。
