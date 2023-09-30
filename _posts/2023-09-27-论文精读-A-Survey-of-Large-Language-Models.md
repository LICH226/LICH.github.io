---
layout: post
title: 论文精读-A Survey of Large Language Models
date: 2023-09-27
Author: YMieMie
tags: [LLM,nlp]
toc: true
comments: true
---

随者大语言模型在NLP相关任务上取得的优越的成果和其普及程度，这篇文章作为综述，从背景、关键的发现还有主流的任务方面来回顾LLm的最新进展。

# Abstract

LLM（ChatGPT）:含有更多参数的PLMs（BERT）

我们特别关注了LLM的四个方面:

1. pre-training:预训练

2. adaptation tuing:适应性调整

3. utilization:利用

4. capacity evaluation:利用性评估

   

摘要关键词：**Large Language Models; Emergent Abilities; Adaptation Tuning; Utilization; Alignment; Capacity Evaluation**

# Introduction

## 语言模型（LM）的研究可分为四个主要的发展阶段：

### Statistical language models (SLM)

基于统计学习方法，基本思想是基于Markov assumption的单词预测模型，例如根据最近的上下文预测下一个单词。

SLM在信息检索上被广泛运用并提高了性能。

遭受到**维度诅咒**，引入backoff estimation and Good–Turing estimation减轻数据稀疏的问题。

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

LLM与PLM有相似的预训练任务和结构，但是有更好的效果，并且有及其惊讶的效果在解决一系列复杂的任务（emergent abilities)  ChatGPT

[![pPbZFRP.png](https://z1.ax1x.com/2023/09/27/pPbZFRP.png)](https://imgse.com/i/pPbZFRP)

## NOWS

聚焦三个主要的差别在LLMs和PLMs之间。

1. LLM具有**emergent**能力（在复杂的任务上），而PLM不具有。

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

调研整理的网站，有相关LLm的[资源](https://github.com/RUCAIBox/LLMSurvey) 





## 章节介绍

Section 2 LLM背景的介绍和GPT模型的演变。

Section 3 总结发展LLMs可用的资源。

Section 4、5、6、7 从上述四个方面回顾和总结了最近的进展。

Section 8 讨论了提示词工程的实用指南

Section 9 回顾LLM的在几个代表领域应用

Section 10 总结发现讨论未来问题。



# Overview

总结LLMs的背景和GPT模型的技术演变。

## Background for LLMs

### Scaling Laws for LLMs

建立量化指标来衡量scaling的影响



**KM scaling law**

model size (N), dataset size (D), and the amount of training compute (C)三个方面**经验**评估



$\Huge L(N) = (\frac{N_c}{N})^{\alpha_N}$                     $\Large \alpha_N \sim 0.076,N_C \sim 8.8 \times 10^{13} $

$\Huge L(D) = (\frac{D_c}{D})^{\alpha_D}$                     $\Large \alpha_D \sim 0.095,D_C \sim 5.4 \times 10^{13} $

$\Huge L(C) = (\frac{C_c}{N})^{\alpha_C}$                       $\Large \alpha_C \sim 0.050,C_C \sim 3.1 \times 10^{13} $



**Chinchilla scaling law**

进行了严格的实验，拟合了数据。



$ \huge  L(N,D) = E +  \frac{A}{M^\alpha} + \frac{B}{D^\beta}$

$\large E = 1.69，A = 406.4, B = 410.7, \alpha = 0.34，\beta = 0.28$ 



优化loss函数，计算最优值：



$\huge N_{opt}(C) = G(\frac{C}{6})^a,D_{opt}(C) = G^{-1}(\frac{C}{6})^b$ 

$ \large a = \frac{\alpha}{\alpha + \beta}，G是由A,B,\alpha,\beta计算$



### Emergent Abilities of LLMs.

“the abilities that are not present in small models but arise in large models“类似物理中的phase transition现象。

介绍三种emergent的典型能力和代表性的模型。

1. ***In-context learning***  GPT3

2. ***Instruction following***  LaMDA-PT 

3. ***Step-by-step reasoning***  chain of thought

   

### Key Techniques for LLMs

帮助大模型成功的几个techniques

1. ***Scaling*** 合理利用compute budget

2. ***Training***  分布式算法 框架 DeepSpeed Megatron-LM

3. ***Ability eliciting***  chain-of-thought prompting

4. ***Alignment tuning***  以人类的values helpful honest 匹配LLM  InstructGPT 是OpenAI 的GPT-3 的后继者模型. 旨在解决用户对GPT-3 的投诉，尤其是有关有毒或误导性输出的投诉。 使用来自人类反馈的强化学习(RLHF) 来增强可靠性和安全性。

5. ***Tools manipulation***  在plain text corpora上表现还不错，在form of text 表现较差；收到预训练数据的限制。用外部工具解决，学会计算器计算和用搜索引擎搜索，装插件。

   
## Technical Evolution of GPT-series Models

OpenAI在 LLm上的研究历程

### Early Explorations

1. ***GPT-1*** :  decoder-only Transformer 确立了基本原则 predicting the next word 无监督预训练+有监督的微调
2. ***GPT-2***： 无监督训练、更多参数、更大数据集，提出了一个"claim"   each (NLP) task can be considered as the word prediction problem based on a subset of the world text

 

### Capacity Leap
***GPT-3*** ：超大尺寸，正式介绍了[ICL](https://zhuanlan.zhihu.com/p/611217770)的概念，大力出奇迹


### Capacity Enhancement

在以下两个方面继续加强：

1. ***training on code data***  缺少对推理能力对于复杂的任务，如代码数学问题。在Codex（GPT3.5）上微调，效果好。
2. ***Human alignment***   Proximal Policy Optimization (PPO) 　RL算法　在GPT-2中先应用。GPT-3用了InstructGPT，建立了three-stage reinforcement learning from human feedback (RLHF) algorithm.


### The Milestones of Language Models

ChatGPT  注重对话

GPT-4  多模态



<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://z1.ax1x.com/2023/09/30/pPq8HhT.md.png" 
         alt="无法显示图片时显示的文字"
         style="zoom:1"/>
    <br>		<!--换行-->
    A timeline of existing large language models (having a size larger than 10B) in recent years<!--标题-->
    </center>
</div>



<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://z1.ax1x.com/2023/09/30/pPq8TA0.md.png"
         alt="无法显示图片时显示的文字"
         style="zoom:1"/>
    <br>		<!--换行-->
    A brief illustration for the technical evolution of GPT-series models.<!--标题-->
    </center>
</div>
# RESOURCES OF LLMS

简要总结了开发大模型的可用资源，包括checkpoints，corpora，libraries

## Publicly Available Model Checkpoints or APIs

根据模型的尺寸分为两个级别（ tens of billions of parameters and hundreds of billions of parameters）

### Models with Tens of Billions of Parameters

1. Flan-T5 (11B version)    基础模型，[instruction tuning](https://zhuanlan.zhihu.com/p/623944861)
2. CodeGen (11B version)  自回归模型，代码生成
3. mT0 (13B version)          多语言任务
4. PanGu-α  擅长中文下游任务，基于MindSpore框架开发

### Models with Hundreds of Billions of Parameters

1. OPT (175B version)  OPT-IML（introduction-version)
2. BLOOM (176B version) and BLOOMZ (176B version)  跨语种任务
3. ChatGLM2-6B (a updated version for ChatGLM-6B ）   中文对话模型

###  LLaMA Model Family

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="https://z1.ax1x.com/2023/09/30/pPq87NV.md.png"
         alt="无法显示图片时显示的文字"
         style="zoom:1"/>
    <br>		<!--换行-->
    An evolutionary graph of the research work conducted on LLaMA.<!--标题-->
    </center>
</div>
 有four sizes (7B, 13B, 30B and 65B)

1. Vicuna 多模态模型
2. Alpaca 开源的instructe模型

### Public API of LLMs

openAI提供了七个主要的接口**ada,babbage, curie, davinci (the most powerful version in GPT-3 series), text-ada-001, text-babbage-001, and text-curie-001**.

各个接口的详细使用都可在[网站](https://platform.openai.com/docs/models/overview
)找到。

## Commonly Used Corpora

分类these corpora into six groups: **Books, CommonCrawl,Reddit links, Wikipedia, Code, and others.**

1. **GPT-3 (175B)** was trained on a mixed dataset of 300B tokens, including CommonCrawl , WebText2,Books1,Books2,and Wikipedia.
2. **PaLM (540B)** uses a pre-training dataset of 780Btokens, which is sourced from social media conversations,filtered webpages, books, Github, multilingual Wikipedia and news.
3. **LLaMA** extracts training data from various sources,including CommonCrawl, C4, Github, Wikipedia,books, ArXiv, and StackExchange. The training data size for LLaMA (6B) and LLaMA (13B) is 1.0T tokens, while 1.4T tokens are used for LLaMA (32B) and LLaMA (65B).

## Library Resource

介绍开发大模型的相关库

1. **Transformers** 提供模型结构的库
2. **DeepSpeed** 深度学习优化库
3. **Megatron-LM**  训练大规模语言模型的库