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

# 1.Introduction

## 1.1 语言模型（LM）的研究可分为四个主要的发展阶段：

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

## 1.2 NOWS

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





## 1.3 章节介绍

Section 2 LLM背景的介绍和GPT模型的演变。

Section 3 总结发展LLMs可用的资源。

Section 4、5、6、7 从上述四个方面回顾和总结了最近的进展。

Section 8 讨论了提示词工程的实用指南

Section 9 回顾LLM的在几个代表领域应用

Section 10 总结发现讨论未来问题。



# 2 Overview

总结LLMs的背景和GPT模型的技术演变。

## 2.1 Background for LLMs

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

   
## 2.2 Technical Evolution of GPT-series Models

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
# 3 RESOURCES OF LLMS

简要总结了开发大模型的可用资源，包括checkpoints，corpora，libraries

## 3.1 Publicly Available Model Checkpoints or APIs

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

## 3.2 Commonly Used Corpora

分类these corpora into six groups: **Books, CommonCrawl,Reddit links, Wikipedia, Code, and others.**

1. **GPT-3 (175B)** was trained on a mixed dataset of 300B tokens, including CommonCrawl , WebText2,Books1,Books2,and Wikipedia.
2. **PaLM (540B)** uses a pre-training dataset of 780Btokens, which is sourced from social media conversations,filtered webpages, books, Github, multilingual Wikipedia and news.
3. **LLaMA** extracts training data from various sources,including CommonCrawl, C4, Github, Wikipedia,books, ArXiv, and StackExchange. The training data size for LLaMA (6B) and LLaMA (13B) is 1.0T tokens, while 1.4T tokens are used for LLaMA (32B) and LLaMA (65B).

## 3.3 Library Resource

介绍开发大模型的相关库

1. **Transformers** 提供模型结构的库
2. **DeepSpeed** 深度学习优化库
3. **Megatron-LM**  训练大规模语言模型的库

# 4 PRE-TRAINING

## 4.1 Data Collection

代表性大模型的数据分布如下：

![image-20231001151914140](C:\Users\15295\AppData\Roaming\Typora\typora-user-images\image-20231001151914140.png)

### 4.1.1 Data source

预训练数据分为两类：general data 和 specialized data

#### General Text Data

1. **Webpages**：网站爬取的文本质量参差不齐，需要作进一步的数据过滤和处理。

2. **Conversation text**：提高对话能力。过多的对话数据会造成对instructions有效性的下降（将陈述句和直接疑问句理解为对话的开始）

3. **Books**：学习语言知识，产生叙事性和连贯性的文本。

#### Specialized Text Data

1. **Multilingual text**: 增强对语言模型的能力，机器翻译、多语言摘要、多语言问答。
2. **Scientific text**:增强对科学文本的理解能力。
3. **Code**:Stack Exchange Github

### 4.1.2 Data Preprocessing

一个典型的预处理数据的流水线图：

[![pPXJcDg.png](https://z1.ax1x.com/2023/10/05/pPXJcDg.png)](https://imgse.com/i/pPXJcDg)

#### Quality Filtering

1. **classified-based**: 可能会过滤一些高质量的数据
2. **heuristic-based**:  Language Filtering、Metric Filtering、Statistic Filtering、Keyword Filtering

#### De-duplication

可在不同的粒度上进行，句子、文档、数据。

#### Privacy Redactio

删除重复数据也可以降低隐私风险。

#### Tokenization

将文本变为单个token序列用作LLm的输入。

[subword tokenizers](https://zhuanlan.zhihu.com/p/620508648) 被广泛应用于transformer模型，包括Encoding tokenization,WordPiece tokenization andUnigram tokenization。

1. **Encoding tokenization** GPT-2, BART, and LLaMA
2. **WordPiece tokenization** BERT
3. **Unigram tokenization** T5 and mBART

### 4.1.3 Effect of Pre-training Data on LLMs

#### Mixture of Sources.

数据多样性，增强在其他领域的泛化性。

#### Amount of Pre-training Data

研究人员在充分训练模型时，特别是在缩放模型参数时，应该更加关注高质量数据的数量。

#### Quality of Pre-training Data

重复的数据可能带来“double descent” (referring to the phenomenon of performance initially deteriorating and subsequently improving)

## 4.2 Architecture

### 4.2.1 Mainstream Architectures

现存的LLM主流架构可分为三个主要类型encoder-decoder, causal decoder, and prefix decoder

[![pPXJ6KS.png](https://z1.ax1x.com/2023/10/05/pPXJ6KS.png)](https://imgse.com/i/pPXJ6KS)

the blue, green, yellow and grey rounded rectangles indicate the attention between prefix tokens, attention between prefix and target tokens, attention between target tokens, and masked attention respectively。



1. **Encoder-decoder Architecture**：编码器：multi-head self-attention，解码器：cross-attention
2. **Causal Decoder Architecture**：unidirectional attention mask
3. **Prefix Decoder Architecture**：允许prefix tokens双向attention ，而对generated tokens单向attention。编码和解码时共享参数，

### 4.2.2 Detailed Configuration（留着看）

#### **Normalization Methods**
LayerNorm，RMSNorm，DeepNorm

#### **Normalization Position**

post-LN, pre-LN, and sandwich-LN

#### **Activation Functions**

GeLU，SwiFGLU，GeGLU

#### **Position Embeddings**

Absolute position embedding（sinusoidal and learned position embeddings）

Relative position embedding（Transformer-XL）

Rotary Position Embedding

ALiBi

#### Attention and Bias

Full attention

Sparse attention（极高计算速度）

Multi-query attention（share the same linear transformation matrices on the keys and values）

FlashAttention（been integrated into PyTorch ，DeepSpeed, and Megatron-LM ）



**总结：For stronger generalization and training stability, it is suggested to choose the pre RMSNorm for layer normalization, and SwiGLU or GeGLU as the activation function. In addition, LN may not be used immediately after embedding layers, which is likely to incur performance degradation.As for position embeddings, RoPE or ALiBi is a better choice since it performs better on long sequences. **

### 4.2.3 Pre-training Tasks

#### language modeling

最通常的目的是仅仅预训练pre-train decoder。

最大化下面的训练目标：

​                                                            $\large  \zeta_{LM}(x) =  \sum_{i=1}^n logP(x_i|x_{<i})$



#### Denoising Autoencoding

恢复被替换的tokens$\widetilde{x} $.

训练目标：

​                                                             $ \large \zeta_{DAE}(x) = logP(\widetilde{x}|x_{\backslash \widetilde{x} }) $

DAE任务似乎比LM任务更复杂，因此还没有广泛的运用于大模型之中。

#### Mixture-of-Denoisers

MoD将LM和DAE视为不同类型的denoising任务，即S-denosier、R-denosier（short span and
low corruption）、X-denoisier（, long span or high corruption）。

S-denosier类似LM，而R-denosier和X-denosier与DAE相似，但spans长度和替换概率不同。

### 4.2.4 Summary and Discussion

#### Architecture Choice

大部分的LLM模型基于casual decoder开发。

1. causal decoder架构能实现更优越的zero-shot和few-shot能力。

2. Scaling law 在casual decoders上被广泛的使用，随着数据和模型参数的增加，casual decoders的性能显著增加。

#### Long Context

基于transformer架构的语言模型会受到文本长度的限制由于二次计算的代价在时间和内村上。

1. Extrapolation LLM需要编码长文本的能力称为extrapolation capability。
2. Efficiency 通过一些方法来降低注意力模块的二次计算成本。

## 4.3 Model Training

### 4.3.1 Optimization Setting

1. **Batch Training**  批量大小的动态调度可以有效地稳定LLms的训练过程。
2. **Learning Rate**  采用线性warmup方法，逐步提高学习率至最大值，然后采用cosine decay方法逐步减小学习率到最大值的大约10%。
3. **Optimizer**  Adam和AdamW优化器被广泛的使用，Adafactor优化器（Adam的一个变体，专门设计用于在训练期间保存GPU内存）也被广泛的应用于训练LLMs
4. **Stabilizing the Training**  weight decay和gradient clipping

[![pPXJgbQ.png](https://z1.ax1x.com/2023/10/05/pPXJgbQ.png)](https://imgse.com/i/pPXJgbQ)


### 4.3.2 Scalable Training Techniques

#### 3D Parallelism

3D Parallelism 是三种并行训练技巧方法的结合，data parallelism, pipeline parallelism , and tensor parallelism。

1. **data parallelism** 每个GPU只需要处理分配给它的数据，并进行前向和后向传播来获得梯度。
2. **pipeline parallelism** 将LLM的不同层分布到多个gpu中。
3. **ensor parallelism** 与管道并行性不同，张量并行性侧重于分解llm的张量(参数矩阵)。



#### ZeRO

关注数据并行中的内存冗余问题，ZeRO技术的目的是在每个GPU上只保留一小部分数据，而其余数据可以在需要时从其他GPU检索。

具体来说，ZeRO根据三部分数据的存储方式提供了三种解决方案，optimizer state partitioning, gradient partitioning, and parameter partitioning。

PyTorch实现了与ZeRO类似的技术，称为FSDP。

#### Mixed Precision Training

近年来，为了预训练超大型语言模型，一些研究已经开始使用16位浮点数(FP16)，这样可以减少内存使用和通信开销。

#### Overall Training Suggestion

在实践中，为了提高训练吞吐量和提高模型的大负载，经常联合使用上述训练技术，尤其是3D parallelism。

# 5 ADAPTATION OF LLMS

经过预训练，LLMs 可以获得解决各种任务的一般能力。然而，越来越多的研究表明，LLMs的能力可以根据具体目标进一步调整。

在本节中，我们将介绍adapting pre-trained llm的两种主要方法，instruction tuning and alignment tuning。

[![pPzXvp6.png](https://z1.ax1x.com/2023/10/11/pPzXvp6.png)](https://imgse.com/i/pPzXvp6)

​                           **A detailed list of available collections for instruction tuning**

## 5.1 Instruction Tuning

从本质上讲，instruction tuning是在自然语言形式的a collection of formatted instances上对预训练的llm进行微调的方法，这与 supervised fine-tuning和multi-task prompted training高度相关。

我们还讨论了使用instruction tuning来满足用户的实际需求，这在现有的llm中已经得到了广泛的应用，例如InstructGPT和GPT-4。

### 5.1.1 Formatted Instance Construction

通常，an instruction-formatted instance由a task description (called an instruction), an optional input, the corresponding output, and a small number of demonstrations (optional)组成。

[![pPzjC0H.png](https://z1.ax1x.com/2023/10/11/pPzjC0H.png)](https://imgse.com/i/pPzjC0H)

1. **Formatting Task Datasets** 有研究表明，指令是llm任务泛化能力的关键因素:通过在标记的数据集上对模型进行微调，去掉task descriptions，会导致模型性能急剧下降。
2. **Formatting Daily Chat Data** 尽管大量的训练实例已经用指令进行了格式化，但它们主要来自公共的NLP数据集，要么缺乏指令多样性，要么与人类的真实需求不匹配。以用户提交的queries作为任务的descriptions。
3. **Formatting Synthetic Data ** 因此，the synthetic method（半自动）是生成大规模llm指令数据的一种有效且经济的方法。
4. **Key Factors for Instance Construction**  Scaling the instructions：增加模型的任务数量可以增强LLM的通用能力，但过多的instances可能会导致模型过拟合，并对影响模型的性能。Formatting design：自然语言格式的设计也高度影响着llm的泛化性能，使用适当数量的范例作为演示，它可以带来实质性的改进



**总结：**

指令的多样性和质量似乎比实例的数量更重要。

邀请标注者编写人类需要的任务比使用特定于数据集的任务更有用，为了减少人工工作，我们可以重用现有的格式化数据集，或者使用现有的llm自动构建指令。



### 5.1.2 Instruction Tuning Strategies

与预训练不同，instruction tuning通常更有效，因为只使用适量的实例进行训练。

对于instruction tuning，有两个重要的方面需要考虑：

1. **instruction tuning**： 由于instruction tuning涉及不同任务的混合，因此在调优期间平衡不同任务的比例非常重要。

2. **Combining Instruction Tuning and Pre-Training**： 在nstruction tuning过程中加入预训练数据，视为模型调优的正则化。

   

### 5.1.3 The Effect of Instruction Tuning

1. **Performance Improvement**： 最近的研究对多个scales(77M到540B)的语言模型进行了实验，结果表明，不同scales的模型都可以从instruction tuning中受益，随着参数尺度的增加，性能得到提高。

2. **instruction tuning**： instruction tuning鼓励模型理解完成任务的自然语言指令，赋予大模型 emergent ability。

3. **Domain Specialization**： instruction tuning是使现有的通用LLMs成为特定领域专家的有效方法（法律、医学、金融、算数计算等）。

   

### 5.1.4 Empirical Analysis for Instruction Tuning

使用不同指令集的微调llm往往会导致在下游任务上具有不同性能的模型变体。在本节中，我们将探讨不同类型的指令在微调llm(即7B LLaMA26)中的效果，并研究几种指令改进策略的有用性。

#### Instruction Datasets

Task-specific instructions（FLAN-T5）、Daily chat instructions（ShareGPT）、Synthetic instructions（Self-Instruct-52K）

#### Improvement Strategies

尽管来自人类用户的真实世界指令更适合微调llm，但很难大规模收集它们，所以主要采用大模型生成的指令。但是这种生成的指令存在一定的问题（poor topic diversity and uneven instruction difficulty (either too simple or too difficult），因此有必要提高合成指令的质量。接下来，我们总结了在现有工作中广泛使用的四种主要改进策略:

1. **Enhancing the instruction complexity**：提高指令的复杂度（adding constraints, increasing reasoning steps, and complicating the input）可以提高llm遵循复杂指令的建模能力。
2. **Increasing the topic diversity**：提高指令数据集的主题多样性有助于激发LLM在现实世界中不同任务上的不同能力。
3. **Scaling the instruction number**：使用更多的指令可以扩展任务知识，提高llm的指令跟随能力
4. **Balancing the instruction difficulty**：由于合成指令往往包含过于简单或过于困难的指令，这很可能导致llm的训练不稳定甚至过拟合。

#### Experimental Setup

实验设置的一些细节。

基于AlpacaFarm评估集对聊天设置进行评估。

对于QA设置，我们选择了两个基准，MMLU和BBH3k (YuLan-Chat发布的BBH基准的一个子集)，并使用启发式规则来分析这些llm的答案，基于它们的默认设置来评估准确性。

#### Results and Analysis

实验代码和数据：https://github.com/RUCAIBox/LLMSurvey/tree/main/Experiments

![pPzj9ne.png](https://z1.ax1x.com/2023/10/11/pPzj9ne.png)

1. Task-formatted instructions 更适合QA设置，但可能不适用于chat设置。
2. 多种指令形式的结合对提高LLMs的理解能力有很大的帮助。
3. 增强指令的复杂性和多样性可以提高模型的性能。
4. 简单地增加指令数量可能并不那么有用，balancing the difficulty也并不总是有用的。



#### **Instruction Tuning Suggestions**

[![pPzXXfx.png](https://z1.ax1x.com/2023/10/11/pPzXXfx.png)](https://imgse.com/i/pPzXXfx)

可以根据表中关于gpu数量和tuning time 的基本统计数据准备计算资源。推荐根据Alpaca的代码来进行instruction tuninghttps://github.com/tatsu-lab/stanford_alpaca/#finetuning。随后，应该选择基本模型并构建指令数据集，正如我们在本节中讨论的那样。

## 5.2 Alignment Tuning

### 5.2.1 Background and Criteria for Alignment

#### Background

这些模型有时可能会表现出意想不到的行为，例如，编造虚假信息，追求不准确的目标，并产生有害的、误导性的和有偏见的表达。

研究表明，alignment可能在一定程度上损害LLMs的一般能力，相关文献将其称**alignment tax**。

#### Alignment Criteria

在这里，我们以三个代表性的alignment criteria(**helpful, honest, and harmless**)为例进行讨论，这些标准在现有文献中被广泛采用。

1. **Helpfulness**：为了提供帮助，LLMs应该展示出一个清晰的尝试，以尽可能简洁有效的方式帮助用户解决他们的任务或回答问题。
2. **Honesty**：在基本层面上，诚实的LLMs应该向用户呈现准确的内容，而不是编造信息。与有益和无害相比，诚实是一个更客观的标准，因此， honesty alignment可能会在减少对人类努力的依赖的情况下得到发展。
3. **Harmlessness**：为了做到无害，它要求LLMs所使用的语言不应带有冒犯性或歧视性。



正如我们所看到的，这些标准是非常主观的，是基于人类的认知而制定的。在现有的工作中，有许多方法可以在aligning LLM时满足这些标准。一种很有前途的技术是**red teaming**，它涉及使用手动或自动手段以对抗的方式探测llm以产生有害输出，然后更新llm以防止此类输出。



### 5.2.2 Collecting Human Feedback

LLMs不能考虑到人类对LLMs输出的主观和定性评价(在本调查中称human feedback)。High-quality human feedback对于使LLMs与人类的偏好和价值观保持一致非常重要。在这一部分中，我们将讨论如何选择一组人工标注人员来收集反馈数据。

#### Human Labeler Selection

在现有的工作中，生成人类反馈数据的主要方法是human annotation。

研究人员评估人类标注者的表现，并选择一组表现良好的human labelers(高一致性)作为**super raters**。

#### Human Feedback Collection

在现有的工作中，主要有三种方法来收集人类标注者的反馈和偏好数据。

1. **Ranking-based approach**：引入了Elo评级系统，通过比较候选输出来得出preference ranking。输出的排序作为训练信号，引导模型偏爱某些输出，从而产生更可靠、更安全的输出。
2. **Question-based approach**：此外，human labelers可以通过回答研究人员设计的某些问题来提供更详细的反馈。**WebGPT**，为了帮助模型过滤和利用检索到的文档中的相关信息，需要人工标记员回答检索到的文档是否对给出的答案有用。
3. **Rule-based approach**：许多研究还开发了基于规则的方法来提供更详细的人类反馈。Sparrow不仅选择了标注者认为最好的response，而且还使用一系列规则来测试模型生成的响应是否满足helpful, correct, and harmless的align标准。

### 5.2.3 Reinforcement Learning from Human Feedback

为了使LLM与人类价值观保持一致，人们提出了reinforcement learning from human feedback(RLHF)，利用收集到的人类反馈数据对LLM进行微调，这有助于提高alignment criteria。

#### RLHF System

RLHF系统主要由三个关键部分组成:a pre-trained LM to be aligned, a reward model learning from human feedback, and a RL algorithm training the LM。

- 具体来说，pre-trained LM通常是一个生成模型，使用现有的预训练的LM参数进行初始化。
-  reward model(RM)提供(学习到的)指导信号，这些信号反映了人类对LM生成的文本的偏好，通常以标量值的形式出现。
- 具体来说，近端策略优化(Proximal Policy Optimization, PPO)是一种在现有工作中广泛使用的RL-align算法。

#### Key Steps for RLHF

[![pPzXz6O.png](https://z1.ax1x.com/2023/10/11/pPzXz6O.png)](https://imgse.com/i/pPzXz6O)

RLHF的总体三步流程：

- **Supervised fine-tuning** 为了使LM最初执行所需的行为，它通常需要收集一个受监督的数据集，其中包含输入提示(指令)和用于微调LM的所需输出。注意，在特定设置或场景中，第一步是可选的。
- **Reward model training** 第二步是使用人类反馈数据来训练RM，RM被训练来预测人类偏好的输出。
- **RL fine-tuning**  在这个步骤中，aligning(即微调)LM被形式化为RL问题。*the pre-trained LM acts as the policy that takes as input a prompt and returns an output text, the action space of it is the vocabulary, the state is the currently generated token sequence, and the reward is provided by the RM*。由于RL算法的不稳定性，最近的研究通过重用具有更高奖励的最佳排名样本，用另一种监督微调取代了RL调优。

### 5.3 Parameter-Efficient Model Adaptation

由于llm由大量的模型参数组成，因此执行全参数调优的成本很高。回顾几个有效的参数调优方法。



#### 5.3.1 Parameter-Efficient Fine-Tuning Methods

在接下来的内容中，我们将简要回顾Transformer语言模型的四种参数高效的微调方法，包括adapter tuning, prefix tuning, prompt tuning and LoRA。

[![pPzjSXD.png](https://z1.ax1x.com/2023/10/11/pPzjSXD.png)](https://imgse.com/i/pPzjSXD)

MHA和FFN分别表示Transformer层的多头注意网络和前馈网络。

- **Adapter Tuning**：适配器调整将小型神经网络模块(称为**adapter**)集成到Transformer模型中。在微调过程中，适配器模块将根据具体的任务目标进行优化，而在此过程中，原始语言模型的参数将被冻结。[预训练模型微调 | 一文带你了解Adapter Tuning - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/574191259)
- **Prefix Tuning**：为了优化prefix vectors，提出了a reparameterization trick，通过学习一个MLP函数，该函数将一个较小的矩阵映射到前缀的参数矩阵，而不是直接优化前缀。优化后的映射函数将被丢弃，只保留派derived prefix vectors，以提高特定任务的性能。[深入理解Prefix Tuning - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/639685912)
- **Prompt Tuning**：与Prefix Tuning不同，Prompt Tuning主要侧重于在输入层加入可训练的prompt vectors 。在实现中， task-specific prompt embeddings与 input text embeddings相结合，然后将其输入到语言模型中。在训练过程中，根据特定任务的监督，只学习prompt embeddings。
- **LoRA (Low-Rank Adaptation)**：LoRA在每个dense layer上施加低秩约束来逼近更新矩阵，从而减少可训练参数以适应下游任务。更新过程可以写成一般形式:**$W←W +∆W$**。LoRA的基本思想是冻结原始矩阵**$W∈\mathbb R^{m×n}$**，同时用低秩分解矩阵来近似参数更新**$∆W$**，即**$∆W = A·B^{T}$**，其中**$A∈\mathbb R^{m×k}$**和**$B∈\mathbb R^{n×k}$**是任务适应的可训练参数，**$k≪min(m, n)$**就是降阶。



#### 5.3.2 Parameter-Efficient Fine-Tuning on LLMs

LoRA已被广泛应用于开源llm(例如LLaMA和BLOOM)，用于参数高效的微调。

作为一个重要的资源，库PEFT(代表参数高效微调)已经在[GitHub](https://github.com/huggingface/peft)上发布。它包括几种广泛使用的高效调优方法，包括LoRA/AdaLoRA、prefixtuning、P-Tuning和prompt-tuning。

此外，它支持许多语言模型，例如GPT-2和LLaMA，并且还涵盖了几个代表性的视觉Transformer模型(例如，ViT和Swin Transformer)。

到目前为止，对于不同的高效调优方法在不同设置或任务下对大型语言模型的影响还缺乏深入的研究。

### 5.4 Memory-Efficient Model Adaptation

#### 5.4.1 Background for Quantization

在神经网络压缩中，量化通常是指从浮点数到整数的映射过程，尤其是8位整数量化。

对于神经网络模型，通常有两种需要量化的数据，即weights (model parameters)和activations (hidden activations)。



为了说明模型量化的基本思想，我们引入一个简单但流行的量化函数(将浮点数$x$转换为量化值$x_q$):

$x_q = R(x/S)−Z$

*In this function, S and Z denote the scaling factor (involving two parameters α and β that determine the clipping range) and zero-point factor (determining symmetric or asymmetric quantization), respectively, and R(·) denotes the rounding operation that maps a scaled floating value to an approximate integer*



#### 5.4.2 Quantization Methods for LLMs

通常有两种主要的模型量化方法，即quantization-aware training (QAT) (requiring additional full model retraining) 和 post-training quantization (PTQ) (requires no model retraining)。

简要回顾几种具有代表性的LLMs的PTQ方法。



**Post-Training Quantization (PTQ)** 

我们首先介绍了LLMs的PTQ方法。

- *Mixed-precision decomposition*  随着模型变大，activations会得到一些数值很大的值（称为outlier），这些outlier主要分布在Transformer层的某些特定特征维度上。基于这一发现，中提出了一种vector-wise方法LLM.int8()，该方法将带有outlier的特征维度与矩阵乘法中的其余维度分开。
- *Fine-grained quantization*  ZeroQuant采用dynamic calibration的token-wise quantization approach来压缩activations。而对于weights(更容易量化)，它使用 group-wise quantization。在实践中，通常使用128的组大小进行模型量化。
- *Balancing the quantization difficulty* 结合了scaling transformation来平衡线性层中权重和激活之间的难度:   $Y = (Xdiag(s)^{−1})·(diag(s)W)$。该公式引入数学等价变换，通过scaling factor **$s$**控制量化难度。
- *Layerwise quantization* 该方法找到最优的quantized weights，使layerwise reconstruction loss最小化:$argmin_{\hat W}||WX-\hat WX||_2^2$。



**Other Quantization Methods**

- *Efficient fine-tuning enhanced quantization* 对于posttraining quantization，direct low-bit quantization(例如，INT4量化)通常会导致很大的性能下降。为了克服这一挑战，QLoRA在量化模型中加入了额外的small tunable adapters(16位精度)，以实现高效、高精度的模型微调。它结合了LoRA和quantization methods的优点。
- *Quantization-aware training (QAT) for LLMs* 最近的一项研究通过应用data-free distillation method来压缩weights, activations以及key-value cache，探讨了QAT方法的效果。



#### 5.4.3 Empirical Analysis and Findings

- *INT8 weight quantization can often yield very good results on LLMs, while the performance of lower precision weight quantization depends on specific methods* 在实践中，在内存成本相同的情况下，建议使用量化精度较低的较大语言模型，而不是量化精度较高的小语言模型。
- *Activations are more difficult to be quantized than weights*  在实践中，high-quality的INT8activation quantization仍然是一项艰巨的任务，尽管有几种方法可以获得令人满意的结果。此外，即使对于QAT方法，也尚未成功探索低精度activation quantization。
- *Efficient fine-tuning enhanced quantization is a good option to enhance the performance of quantized LLMs* 



#### 5.4.4 Open-source Libraries and Quantized LLMs

**Quantization Libraries** 

- *Bitsandbytes34* 重点研究llm的INT8量化，主要提供对8位矩阵乘法和8位优化器的支持。
- *GPTQ-for-LLaMA* 是专门为量化LLaMA模型而开发的。
- *AutoGPTQ* 是基于GPTQ算法开发的量化包[293]，支持llm的INT4量化。
- *llama.cpp* 它支持INT4、INT5和INT8量化，采用高效的C/C+语言开发。

**Quantized LLMs**

与原始模型相比，quantized language models占用的内存更小，并且可能具有更快的推理速度。

可以在HuggingFace上找到相关的量化好的模型。
