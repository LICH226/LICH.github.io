---
layout: post
title: Python中的Pathlib
date: 2023-08-08
Author: YMieMie
tags: [ Python]
toc: true
comments: true
---

学习使用python中的Pathlib库来代替os处理文件/文件夹/路径等。使用Google下的Colab环境。

# os和Pathlib比较

**`os.path` 的最大缺点是将系统路径视为字符串，极容易导致混乱**，`Pathlib` 在Python3.4中被支持， 通过将路径表示为独特的对象解决了这个问题，并为路径处理引入更多可扩展用法，许多操作在`os`需要层层嵌套，而`Pathlib`将使开发人员更轻松地处理与路径和文件相关的所有事情。

# 处理路径

## 1.创建路径

几乎所有`pathlib` 的功能都可以通过其 `Path` 子类访问，可以使用该类创建文件和目录，有多种初始化`Path`的方式。

使用当前工作路径：

```python
from pathlib import Path

Path.cwd() #PosixPath('/content')
```

使用home：

```python
Path.home() # PosixPath('/root')
```

同样的可以指定字符串路径创建路径：

```python
p = Path("documents") # PosixPath('documents')
```

使用**正斜杠**运算符进行路径连接：

```python
data_dir = Path(".") / "data"
csv_file = data_dir / "file.csv"
print(data_dir) # data
print(csv_file) # data/file.csv
```

检查路径是否存在，可以使用布尔函数 `exists`

```python
data_dir.esists() # false
```

使用 `is_dir` 或 `is_file` 函数来检查是否为文件夹、文件

```python
data_dir.is_dir()
csv_file.is_file()
```

大多数路径都与当前运行目录相关，但某些情况下必须提供文件或目录的**绝对路径**，可以使用 `absolute`:

```python
csv_file.absolute() # PosixPath('/content/data/file.csv')
```

如果仍然需要将路径转为字符串，可以调用 `str(path)` 强制转换:

```python
str(Path.home()) # '/root'
```

现如今大多数库都支持 `Path` 对象，包括 `sklearn` 、 `pandas` 、 `matplotlib` 、 `seaborn` 等

## 2. Path属性

`Path` 对象有许多有用属性，一起来看看这些示例，首先定义一个图片路径

```python
image_file = Path("images/shadousheng.png").absolute() 
# PosixPath('/content/images/shadousheng.png')
```

`parent` ，它将返回**当前工作目录的上一级**

```python
image_file.parent # PosixPath('/content/images')
```

获取文件名

```python
image_file.name # 'shadousheng.png'
```

它将返回带有后缀的文件名，**若只想要前缀**，则使用`stem`

```python
image_file.stem # shadousheng
```

只想要后缀也很简单

```python
image_file.suffix # '.png'
```

如果要将路径分成多个部分，可以使用 `parts`

```python
image_file.parts # ('/', 'content', 'images', 'shadousheng.png')
```

如果希望这些组件本身就是 `Path` 对象，可以使用 `parents` 属性，它会创建一个生成器

```python
for i in image_file.parents:
    print(i)

# /content/images
# /content
# /
```

# 处理文件

想要创建文件并写入内容，不必再使用 `open` 函数，只需创建一个 `Path` 对象搭配 `write_text` 或 `write_btyes` 即可

```python
markdown = data_dir / "file.md"

# Create (override) and write text
markdown.write_text("# This is a test markdown")
```

读取文件，可以 `read_text` 或 `read_bytes`

```python
markdown.read_text() # '# This is a test markdown'
```

```python
len(image_file.read_bytes()) # 1962148
```

但请注意， `write_text` 或 `write_bytes` 会覆盖文件的现有内容

```python
# Write new text to existing file
markdown.write_text("## This is a new line")
```

```python
# The file is overridden
markdown.read_text() # '## This is a new line'
```

要将新信息附加到现有文件，应该在 `a` （附加）模式下使用 `Path` 对象的 `open` 方法：

```python
# Append text
with markdown.open(mode="a") as file:
    file.write("\n### This is the second line")

markdown.read_text() # '## This is a new line\n### This is the second line'
```

使用`rename` 重命名文件，如在当前目录中重命名，`file.md` 变成了 `new_markdown.md`

```python
renamed_md = markdown.with_stem("new_markdown") # PosixPath('data/new_markdown.md')

markdown.rename(renamed_md) # PosixPath('data/new_markdown.md')
```

通过 `stat().st_size` 查看文件大小

```python
# Display file size
renamed_md.stat().st_size # 49
```

查看最后一次修改文件的时间

```python
from datetime import datetime

modified_timestamp = renamed_md.stat().st_mtime

datetime.fromtimestamp(modified_timestamp) # datetime.datetime(2023, 8, 1, 13, 32, 45, 542693)
```

`st_mtime` 返回一个自 1970 年 1 月 1 日以来的秒数。为了使其可读，搭配使用 `datatime` 的 `fromtimestamp` 函数。

要删除不需要的文件，可以 `unlink`

```python
renamed_md.unlink(missing_ok=True)
```

如果文件不存在，将 `missing_ok` 设置为 `True` 则不会引起报错

# 处理目录

首先，看看如何递归创建目录

```python
new_dir = Path.cwd() / "data" / "new_dir"
new_dir.mkdir(parents=True, exist_ok=True)
```

默认情况下， `mkdir` **创建给定路径的最后一个子目录**，如果中间父级不存在，则必须将 `parents` 设置为 `True` 达到递归创建目的

要删除目录，可以使用 `rmdir` ，**如果给定的路径对象是嵌套的，则仅删除最后一个子目录**

```python
new_dir.rmdir()
```

要在终端上列出 `ls` 等目录的内容，可以使用 `iterdir` 。结果将是一个生成器对象，一次生成一个子内容作为单独的路径对象，和`os.listdir`不同的是，**它返回每个内容的绝对路径而不是名字**

```python
for p in Path.home().iterdir():
    print(p)
# /root/.bashrc
# /root/.profile
# /root/.local
# /root/.jupyter
# /root/.tmux.conf
# /root/.npm
# /root/.config
# /root/.cache
# /root/.wget-hsts
# /root/.ipython
# /root/.keras
# /root/.launchpadlib
```

要捕获具有特定扩展名或名称的所有文件，可以将 `glob` 函数与正则表达式结合使用。

例如，使用 `glob("*.txt")` 查找主目录中所有文本文件

```python
home = Path.home()
text_files = list(home.glob("*.txt"))

len(text_files) # 0
```

要递归搜索文本文件（即在所有子目录中），可以glob 与 `rglob` 结合使用：

```python
all_text_files = [p for p in home.rglob("*.txt")]

len(all_text_files) # 0
```

