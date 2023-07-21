---
layout: post
title: NLP赛事实践
date: 2023-07-21
Author: YMieMie
tags: [NLP, document]
comments: true
---
# DataWhale《AI夏令营-NLP赛事实践》

这篇博客用来记录datawhale暑期夏令营NLP赛事实践的笔记和心得代码等。

## 赛题解析  基于论文摘要的文本分类与关键词抽取挑战赛

赛题链接：https://challenge.xfyun.cn/topic/info?type=abstract-of-the-paper&ch=ZuoaKcY

### 实践任务

任务1：从论文标题、摘要作者等信息，判断该论文是否属于医学领域的文献。

任务2：从论文标题、摘要作者等信息，提取出该论文关键词。

### 数据解析

训练集与测试集数据为CSV格式文件，各字段分别是标题、作者和摘要。Keywords为任务2的标签，label为任务1的标签。训练集和测试集都可以通过pandas读取。

```python
train = pd.read_csv('/home/aistudio/data/data231041/train.csv')
train.head(5)
```

| uuid | title |                      author                       |                     abstract                      |                     Keywords                      |                       label                       | text |                                                   |
| :--: | :---: | :-----------------------------------------------: | :-----------------------------------------------: | :-----------------------------------------------: | :-----------------------------------------------: | :--: | :-----------------------------------------------: |
|  0   |   0   | Accessible Visual Artworks for Blind and Visua... | Quero, Luis Cavazos; Bartolome, Jorge Iranzo; ... | Despite the use of tactile graphics and audio ... | accessibility technology; multimodal interacti... |  0   | Accessible Visual Artworks for Blind and Visua... |
|  1   |   1   | Seizure Detection and Prediction by Parallel M... | Li, Chenqi; Lammie, Corey; Dong, Xuening; Amir... | During the past two decades, epileptic seizure... | CNN; Seizure Detection; Seizure Prediction; EE... |  1   | Seizure Detection and Prediction by Parallel M... |
|  2   |   2   | Fast ScanNet: Fast and Dense Analysis of Multi... | Lin, Huangjing; Chen, Hao; Graham, Simon; Dou,... | Lymph node metastasis is one of the most impor... | Histopathology image analysis; computational p... |  1   | Fast ScanNet: Fast and Dense Analysis of Multi... |
|  3   |   3   | Long-Term Effectiveness of Antiretroviral Ther... | Huang, Peng; Tan, Jingguang; Ma, Wenzhe; Zheng... | In order to assess the effectiveness of the Ch... | HIV; ART; mortality; observational cohort stud... |  0   | Long-Term Effectiveness of Antiretroviral Ther... |
|  4   |   4   | Real-Time Facial Affective Computing on Mobile... | Guo, Yuanyuan; Xia, Yifan; Wang, Jing; Yu, Hui... | Convolutional Neural Networks (CNNs) have beco... | facial affective computing; convolutional neur... |  0   | Real-Time Facial Affective Computing on Mobile... |

## Baseline版本

### 实践思路

#### 任务一

针对文本分类任务，可以提供两种实践思路，一种是使用传统的特征提取方法（如TF-IDF/BOW）结合机器学习模型，另一种是使用预训练的BERT模型进行建模。使用特征提取 + 机器学习的思路步骤如下：

1. 数据预处理：首先，对文本数据进行预处理，包括文本清洗（如去除特殊字符、标点符号）、分词等操作。可以使用常见的NLP工具包（如NLTK或spaCy）来辅助进行预处理。
2. 特征提取：使用TF-IDF（词频-逆文档频率）或BOW（词袋模型）方法将文本转换为向量表示。TF-IDF可以计算文本中词语的重要性，而BOW则简单地统计每个词语在文本中的出现次数。可以使用scikit-learn库的TfidfVectorizer或CountVectorizer来实现特征提取。
3. 构建训练集和测试集：将预处理后的文本数据分割为训练集和测试集，确保数据集的样本分布均匀。
4. 选择机器学习模型：根据实际情况选择适合的机器学习模型，如朴素贝叶斯、支持向量机（SVM）、随机森林等。这些模型在文本分类任务中表现良好。可以使用scikit-learn库中相应的分类器进行模型训练和评估。
5. 模型训练和评估：使用训练集对选定的机器学习模型进行训练，然后使用测试集进行评估。评估指标可以选择准确率、精确率、召回率、F1值等。
6. 调参优化：如果模型效果不理想，可以尝试调整特征提取的参数（如词频阈值、词袋大小等）或机器学习模型的参数，以获得更好的性能。

#### 代码思路(在原文档修改)

TfidfVectorizer 参数及属性的最详细解析 [TfidfVectorizer 参数及属性的最详细解析 | 程序员笔记 (knowledgedict.com)](https://www.knowledgedict.com/tutorial/sklearn-tfidfvectorizer-parameters-and-attributes.html)

```python
# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型)TfidfVectorizer（TF-IDF（词频-逆文档频率））
from sklearn.feature_extraction.text import TfidfVectorizer

# 导入LogisticRegression回归模型
from sklearn.linear_model import LogisticRegression

# 过滤警告消息
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# 读取数据集
train = pd.read_csv('/home/aistudio/data/data231041/train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

test = pd.read_csv('/home/aistudio/data/data231041/test.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')


# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')

vector = TfidfVectorizer(ngram_range=(1,10),max_features=3000).fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])


# 引入模型
model = LogisticRegression()

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)

# 生成任务一推测结果
test[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)
```

