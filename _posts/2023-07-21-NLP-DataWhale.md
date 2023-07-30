---
layout: post
title: NLP赛事实践
date: 2023-07-21
Author: YMieMie
tags: [NLP, document]
comments: true
---
这篇博客用来记录datawhale暑期夏令营NLP赛事实践的笔记和心得代码等。
# DataWhale《AI夏令营-NLP赛事实践》

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

## BERT版本

本版本使用BERT进行微调。

```python
#导入前置依赖
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# 用于加载bert模型的分词器
from transformers import AutoTokenizer
# 用于加载bert模型
from transformers import BertModel
from pathlib import Path
```

```python
#导入模型参数
batch_size = 16
text_max_length = 128
epochs = 100
lr = 3e-5
validation_ratio = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_per_step = 50


model_dir = Path("./model/bert_checkpoints")
os.makedirs(model_dir) if not os.path.exists(model_dir) else ''
print("Device:",device)
```

```markdown
Device: cuda
```

```python
#读取数据
pd_train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
pd_train_data["title"] = pd_train_data["title"].fillna("")
pd_train_data["abstract"] = pd_train_data["abstract"].fillna("")
test_data['title'] = test_data['title'].fillna('')
test_data['abstract'] = test_data['abstract'].fillna('')
pd_train_data['text'] = pd_train_data['title'].fillna('') + ' ' +  pd_train_data['author'].fillna('') + ' ' + pd_train_data['abstract'].fillna('')+ ' ' + pd_train_data['Keywords'].fillna('')
test_data['text'] = test_data['title'].fillna('') + ' ' +  test_data['author'].fillna('') + ' ' + test_data['abstract'].fillna('')+ ' ' + pd_train_data['Keywords'].fillna('')
```

```python
#从训练集中随机采样测试集
validation_data = pd_train_data.sample(frac=validation_ratio)
train_data = pd_train_data[~pd_train_data.index.isin(validation_data.index)]
```

```python
len(validation_data),len(train_data)
```

```markdown
(600, 5400)
```

```python
# 构建Dataset
class MyDataset(Dataset):

    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode
        # 拿到对应的数据
        if mode == 'train':
            self.dataset = train_data
        elif mode == 'validation':
            self.dataset = validation_data
        elif mode == 'test':
            # 如果是测试模式，则返回内容和uuid。拿uuid做target主要是方便后面写入结果。
            self.dataset = test_data
        else:
            raise Exception("Unknown mode {}".format(mode))

    def __getitem__(self, index):
        # 取第index条
        data = self.dataset.iloc[index]
        # 取其内容
        text = data['text']
        # 根据状态返回内容
        if self.mode == 'test':
            # 如果是test，将uuid做为target
            label = data['uuid']
        else:
            label = data['label']
        # 返回内容和label
        return text, label

    def __len__(self):
        return len(self.dataset)

```

```python
#构建数据集
train_dataset = MyDataset('train')
validation_dataset = MyDataset('validation')
```

```python
train_dataset[0],validation_dataset[0]
```

```markdown
(('Accessible Visual Artworks for Blind and Visually Impaired People: Comparing a Multimodal Approach with Tactile Graphics Quero, Luis Cavazos; Bartolome, Jorge Iranzo; Cho, Jundong Despite the use of tactile graphics and audio guides, blind and visually impaired people still face challenges to experience and understand visual artworks independently at art exhibitions. Art museums and other art places are increasingly exploring the use of interactive guides to make their collections more accessible. In this work, we describe our approach to an interactive multimodal guide prototype that uses audio and tactile modalities to improve the autonomous access to information and experience of visual artworks. The prototype is composed of a touch-sensitive 2.5D artwork relief model that can be freely explored by touch. Users can access localized verbal descriptions and audio by performing touch gestures on the surface while listening to themed background music along. We present the design requirements derived from a formative study realized with the help of eight blind and visually impaired participants, art museum and gallery staff, and artists. We extended the formative study by organizing two accessible art exhibitions. There, eighteen participants evaluated and compared multimodal and tactile graphic accessible exhibits. Results from a usability survey indicate that our multimodal approach is simple, easy to use, and improves confidence and independence when exploring visual artworks. accessibility technology; multimodal interaction; auditory interface; touch interface; vision impairment',
  0),
 ('Probing the origin of prion protein misfolding via reconstruction of ancestral proteins Leonardo M Cortez,Anneliese J Morrison,Craig R Garen,Sawyer Patterson,Toshi Uyesugi,Rafayel Petrosyan,Rohith Vedhthaanth Sekar,Michael J Harms,Michael T Woodside,Valerie L Sim,Leonardo M Cortez,Anneliese J Morrison,Craig R Garen,Sawyer Patterson,Toshi Uyesugi,Rafayel Petrosyan,Rohith Vedhthaanth Sekar,Michael J Harms,Michael T Woodside,Valerie L Sim Prion diseases are fatal neurodegenerative diseases caused by pathogenic misfolding of the prion protein, PrP. They are transmissible between hosts, and sometimes between different species, as with transmission of bovine spongiform encephalopathy to humans. Although PrP is found in a wide range of vertebrates, prion diseases are seen only in certain mammals, suggesting that infectious misfolding was a recent evolutionary development. To explore when PrP acquired the ability to misfold infectiously, we reconstructed the sequences of ancestral versions of PrP from the last common primate, primate-rodent, artiodactyl, placental, bird, and amniote. Recombinant ancestral PrPs were then tested for their ability to form β-sheet aggregates, either spontaneously or when seeded with infectious prion strains from human, cervid, or rodent species. The ability to aggregate developed after the oldest ancestor (last common amniote), and aggregation capabilities diverged along evolutionary pathways consistent with modern-day susceptibilities. Ancestral bird PrP could not be seeded with modern-day prions, just as modern-day birds are resistant to prion disease. Computational modeling of structures suggested that differences in helix 2 could account for the resistance of ancestral bird PrP to seeding. Interestingly, ancestral primate PrP could be converted by all prion seeds, including both human and cervid prions, raising the possibility that species descended from an ancestral primate have retained the susceptibility to conversion by cervid prions. More generally, the results suggest that susceptibility to prion disease emerged prior to ~100 million years ago, with placental mammals possibly being generally susceptible to disease. ancestral sequence reconstruction; prion protein; protein aggregation; seeding; structural modeling.',
  1))
```

```python
#获取Bert预训练模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

```python
#接着构造我们的Dataloader。
#我们需要定义一下collate_fn，在其中完成对句子进行编码、填充、组装batch等动作：
def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果，例如：
             src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
             target：[1, 1, 0, ...]
    """
    text, label = zip(*batch)
    text, label = list(text), list(label)

    # src是要送给bert的，所以不需要特殊处理，直接用tokenizer的结果即可
    # padding='max_length' 不够长度的进行填充
    # truncation=True 长度过长的进行裁剪
    src = tokenizer(text, padding='max_length', max_length=text_max_length, return_tensors='pt', truncation=True)

    return src, torch.LongTensor(label)
```

```python
#构建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
```

```python
inputs, targets = next(iter(train_loader))
print("inputs:", inputs)
print("targets:", targets)
```

```markdown
inputs: {'input_ids': tensor([[  101,  2784, 17841,  ...,  1996, 18921,   102],
        [  101,  1037,  7132,  ...,  2000, 12826,   102],
        [  101, 21933, 10626,  ...,  1006, 17804,   102],
        ...,
        [  101,  3161,  9530,  ..., 21163,  2944,   102],
        [  101,  6047,  4942,  ...,  2024,  3651,   102],
        [  101,  7667,  1997,  ...,  1996,  2909,   102]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        ...,
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1]])}
targets: tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1])
```

```python
#看看bert的输出
bert = BertModel.from_pretrained('bert-base-uncased')
output = bert(**inputs)
output.last_hidden_state.shape
```

```python
torch.Size([16, 128, 768])
```

```python
#定义预测模型，该模型由bert模型加上最后的预测层组成
class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        # 加载bert模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 最后的预测层
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, src):
        """
        :param src: 分词后的推文数据
        """

        # 将src直接序列解包传入bert，因为bert和tokenizer是一套的，所以可以这么做。
        # 得到encoder的输出，用最前面[CLS]的输出作为最终线性层的输入
        outputs = self.bert(**src).last_hidden_state[:, 0, :]

        # 使用线性层来做最终的预测
        return self.predictor(outputs)

```

```python
model = MyModel()
model = model.to(device)
```

```markdown
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```

上面的输出就是说，bert的与预训练模型不能直接用来做文章的二分类，需要对该预训练模型进行微调。

```python
#定义出损失函数和优化器。这里使用Binary Cross Entropy：
criteria = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

```python
# 由于inputs是字典类型的，定义一个辅助函数帮助to(device)
def to_device(dict_tensors):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors
```

```python
#进行一次简单的测试
model.eval()
total_loss = 0.
total_correct = 0
for inputs, targets in validation_loader:
    inputs, targets = to_device(inputs), targets.to(device)
    outputs = model(inputs)
    print(outputs,outputs.shape)
    loss = criteria(outputs.view(-1), targets.float())
    total_loss += float(loss)

    correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
    total_correct += correct_num
    break
print(total_correct,total_loss)
```

```markdown
tensor([[0.4160],
        [0.4424],
        [0.4082],
        [0.3988],
        [0.3816],
        [0.4362],
        [0.4071],
        [0.4286],
        [0.3975],
        [0.3995],
        [0.4136],
        [0.3880],
        [0.4153],
        [0.4049],
        [0.4047],
        [0.4367]], device='cuda:0', grad_fn=<SigmoidBackward0>) torch.Size([16, 1])
tensor(7, device='cuda:0') 0.710857629776001
```

```python
#定义一个验证方法，获取到验证集的精准率和loss
def validate():
    model.eval()
    total_loss = 0.
    total_correct = 0
    for inputs, targets in validation_loader:
        inputs, targets = to_device(inputs), targets.to(device)
        outputs = model(inputs)
        loss = criteria(outputs.view(-1), targets.float())
        total_loss += float(loss)

        correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
        total_correct += correct_num

    return total_correct / len(validation_dataset), total_loss / len(validation_dataset)
```

```python
# 首先将模型调成训练模式
model.train()

# 清空一下cuda缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 定义几个变量，帮助打印loss
total_loss = 0.
# 记录步数
step = 0

# 记录在验证集上最好的准确率
best_accuracy = 0

# 开始训练
for epoch in range(epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        # 从batch中拿到训练数据
        inputs, targets = to_device(inputs), targets.to(device)
        # 传入模型进行前向传递
        outputs = model(inputs)
        # 计算损失
        loss = criteria(outputs.view(-1), targets.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += float(loss)
        step += 1

        if step % log_per_step == 0:
            print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch+1, epochs, i, len(train_loader), total_loss))
            total_loss = 0

        del inputs, targets

    # 一个epoch后，使用过验证集进行验证
    accuracy, validation_loss = validate()
    print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch+1, accuracy, validation_loss))
    torch.save(model, model_dir / f"model_{epoch}.pt")

    # 保存最好的模型
    if accuracy > best_accuracy:
        torch.save(model, model_dir / f"model_best.pt")
        best_accuracy = accuracy
```

```markdown
Epoch 1/100, Step: 49/338, total loss:4.0771
Epoch 1/100, Step: 99/338, total loss:3.1354
Epoch 1/100, Step: 149/338, total loss:3.8202
Epoch 1/100, Step: 199/338, total loss:4.8310
Epoch 1/100, Step: 249/338, total loss:4.0593
Epoch 1/100, Step: 299/338, total loss:3.3685
Epoch 1, accuracy: 0.9667, validation loss: 0.0065
Epoch 2/100, Step: 11/338, total loss:2.9033
Epoch 2/100, Step: 61/338, total loss:2.2871
Epoch 2/100, Step: 111/338, total loss:2.5146
Epoch 2/100, Step: 161/338, total loss:1.8886
Epoch 2/100, Step: 211/338, total loss:2.9612
Epoch 2/100, Step: 261/338, total loss:4.0673
Epoch 2/100, Step: 311/338, total loss:3.4182
Epoch 2, accuracy: 0.9617, validation loss: 0.0063
Epoch 3/100, Step: 23/338, total loss:2.8668
Epoch 3/100, Step: 73/338, total loss:1.8860
Epoch 3/100, Step: 123/338, total loss:1.9490
Epoch 3/100, Step: 173/338, total loss:2.1700
Epoch 3/100, Step: 223/338, total loss:1.5563
Epoch 3/100, Step: 273/338, total loss:1.3872
Epoch 3/100, Step: 323/338, total loss:1.4699
Epoch 3, accuracy: 0.9700, validation loss: 0.0077
Epoch 4/100, Step: 35/338, total loss:1.5005
Epoch 4/100, Step: 85/338, total loss:2.4052
...
Epoch 100/100, Step: 237/338, total loss:34.6206
Epoch 100/100, Step: 287/338, total loss:34.8460
Epoch 100/100, Step: 337/338, total loss:34.6774
Epoch 100, accuracy: 0.5300, validation loss: 0.0438
```

```python
#加载最好的模型，然后进行测试集的预测
model = torch.load(model_dir / f"model_best.pt")
model = model.eval()
```

```python
#加载测试集
test_dataset = MyDataset('test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
```

```python
#运行模型来预测
results = []
for inputs, ids in test_loader:
    outputs = model(inputs.to(device))
    outputs = (outputs >= 0.5).int().flatten().tolist()
    ids = ids.tolist()
    results = results + [(id, result) for result, id in zip(outputs, ids)]
```

```python
#生成提交的文件
test_label = [pair[1] for pair in results]
test_data['label'] = test_label
test_data[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)
```

