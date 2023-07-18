---
layout: post
title: 依存分析简介
date: 2023-07-18
author: YMieMie
tags: [nlp, Dependency]
comments: true
toc: true
pinned: true
---

一篇关于依存分析的笔记


# Dependency Parsing 依存分析

课程链接：[百度AI Studio课程_学习成就梦想，AI遇见未来_AI课程 - 百度AI Studio - 人工智能学习与实训社区 (baidu.com)](https://aistudio.baidu.com/aistudio/education/lessonvideo/1000411)

## 依存分析简介

依存分析在意的是两个词汇之间的关系，两个词汇不一定要求相连，这一点与选取分析不同。

[![pCTGmRS.md.png](https://s1.ax1x.com/2023/07/18/pCTGmRS.md.png)](https://imgse.com/i/pCTGmRS)

***eg：定票-->飞机票***



依存分析就是把一个句子变成一个图，图上有节点和边。

依存关系由**核心词**（head）与**依存词**（dependent）表示，每个核心词对应其成分的中心（如名词之于名词短语，动词之于动词短语）。最常用的关系分为两大类：**从句关系**（clausal relations）与**修饰语关系**（modiﬁer relations）。

例如，「趣味」是「脱离」的 DOBJ 关系，也就是**直接宾语**（Direct object），这就是从句关系；「低级」是「趣味」的 AMOD 关系，也就是**形容词修饰语**（Adjectival modiﬁer），这就是修饰语关系。

[![pCTGKMQ.md.png](https://s1.ax1x.com/2023/07/18/pCTGKMQ.md.png)](https://imgse.com/i/pCTGKMQ)

***root会被指向没有被任何一个词汇指到的node，也是这个句子最关键的词汇。***

***每个词汇只会有一个边指进来，除了root***。

***每一个node都有唯一的路径回溯到root。***

[![pCTGnxg.md.png](https://s1.ax1x.com/2023/07/18/pCTGnxg.md.png)](https://imgse.com/i/pCTGnxg)

依存分析关系表

[![pCTG8I0.md.png](https://s1.ax1x.com/2023/07/18/pCTG8I0.md.png)](https://imgse.com/i/pCTG8I0)](https://imgse.com/i/pCTGKMQ)



## 模型构建

依存分析模型与成分分析模型很相似，也是通过一个二分类来分类两个node是否具有edge，还有一个多分类来分类改edge所属与的关系。

***tips：$W_l$和$W_r$可以互换，word可以在左侧输入也可以在右侧输入，结果不同。***

[![pCTGQqs.png](https://s1.ax1x.com/2023/07/18/pCTGQqs.png)](https://imgse.com/i/pCTGQqs)

## 模型问题和改进

由于每个分类器是独立的，可能发生如下情况：

[![pCTG3aq.md.png](https://s1.ax1x.com/2023/07/18/pCTG3aq.md.png)](https://imgse.com/i/pCTG3aq)

而每一个node只能有一个被一个node指向，因此这个tree不符合实际。

### 解决方法 Maximum Spanning Tree 

分类器会输出，两两之间node的形成edge的分数，然后计算每一种case得分数谁最高

[![pCTG1Zn.md.png](https://s1.ax1x.com/2023/07/18/pCTG1Zn.md.png)](https://imgse.com/i/pCTG1Zn)

