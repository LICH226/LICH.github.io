---
layout: post
title: 论文精读-A Survey of Large Language Models
date: 2023-09-27
Author: YMieMie
tags: [LLMr,nlp]
toc: true
comments: true
---

随者大语言模型在NLP相关任务上取得的优越的成果和其普及程度，这篇文章作为综述，从背景、关键的发现还有主流的任务方面来回顾LLm的最新进展。

# Abstract

LLM（ChatGPT）:含有更多参数的PLMs（BERT）

我们特别关注了LLM的四个方面:

1. pre-training:预训练

2. adaptation tuing:适应性调整

3. utilization：利用

4. capacity evaluation：利用性评估

   

摘要关键词：Large Language Models; Emergent Abilities; Adaptation Tuning; Utilization; Alignment; Capacity Evaluation

# Introduction

## 语言模型（LM）的研究可分为四个主要的发展阶段：

### Statistical language models (SLM)

基于统计学习方法，基本思想是基于Markov assumption的单词预测模型，例如根据最近的上下文预测下一个单词。

SLM在信息检索上被广泛运用并提高了性能。

遭受到维度诅咒，引入backoff estimation and Good–Turing estimation减轻数据稀疏的问题。

### Neural language models (NLM)

基于神经网络预测，例如RNN。

词的分布式表示，聚合上下文特征（词向量）的预测。

word2vector提供了简单的浅层神经网络来学习distributed word representations

### Pre-trained language models (PLM)

早期的尝试，ELMo通过预训练一个双向的biLSTM，然后根据下游任务微调biLSTM。

Further，BERT（基于自注意力机制和高度并行化的Transformer），提高了语义表示特征的效果。

提供了pre-trained 和 fine-tuning的 paradigm.

### Large language models (LLM)

LLM:更大的模型参数或数据量

LLM与PLM有相似的预训练任务和结构，但是有更好的效果，并且有及其惊讶的效果在解决一系列复杂的任务（emergent abilities)

ChatGPT

[![pPbZFRP.png](https://z1.ax1x.com/2023/09/27/pPbZFRP.png)](https://imgse.com/i/pPbZFRP)

## NOWS

聚焦三个主要的差别在LLMs和PLMs之间。

1. LLM具有emergent能力（在复杂的任务上），而PLM不具有。

2. LLM革新人们使用AI和发展AI的方式。人们必须理解LLMs的工作，然后以LLMs的方式规范行为。

3. LLM不再明确区分research和engineering。

   

LLMs导致了通用人工智能的重新思考（artificial general intelligence（AGI））

IR领域，ChatGPT和New Bing机器对话形式challenged传统的搜索引擎。

CV领域，GPT4支持多模态。



LLM的基础principles深入的研究探索：

1. emergent abilities能力的突然出现。
2. 研究界很难训练这种LLMs。
3. 与人类价值和偏好结合。



对LLM的四个研究方面：

1. pre-training (how to pretrain a capable LLM)
2.  adaptation (how to effectively adapt pre-trained LLMs for better use)
3. utilization (how to use LLMs for solving various downstream tasks)
4. capability evaluation (how to evaluate the abilities of LLMs and existing empirical findings)

调研整理的网站，有相关LLm的资源https://github.com/RUCAIBox/LLMSurvey





## 章节介绍

Section 2 LLM背景的介绍和GPT模型的演变。

Section 3 总结发展LLMs可用的资源。

Section 4、5、6、7 从上述四个方面回顾和总结了最近的进展。

Section 8 讨论了提示词工程的实用指南

Section 9 回顾LLM的在几个代表领域应用

Section 10 总结发现讨论未来问题。
