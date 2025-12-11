# Task01 学习概览

## 1 自然语言处理

- 应用范围：网络搜索、广告、电子邮件、智能客服、机器翻译、智能新闻播报等
- 课程目的：
  1. 基于最前沿的深度学习模型结构（Transformers）解决 NLP 中的经典任务
  2. 了解 Transformer 相关原理
  3. 熟练使用 Transformer 相关的深度学习模型

## 2 常见的 NLP 任务

- 文本分类：对单个或者两个以上的多段文本进行分类
- 序列标注：对文本序列中的 token、字或者词进行分类
- 问答任务
  1. 抽取式问答：根据问题从一段给定的文本中找打答案，答案必须是给定文本的一小段文字。
  2. 多选问答：从多个选项中选择一个正确答案
- 生成任务
  1. 语言模型：根据已有的一段文字生成一个字
  2. 摘要生成：根据一大段文字生成一小段总结性文字
  3. 机器翻译：将源语言翻译成目标语言

## 3 Transformer 兴起

- Transformer 模型来源：2017 年的 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 论文，提出 Transformer 模型结构，并在机器翻译任务上取得了 SOTA 的效果
- 进展：2018 年的论文 [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)，使用 Transformer 模型结构进行大规模语言模型预训练，在多个 NLP 下游任务中进行微调
- 改进：2019~2021 年期间，将 Transformer 模型结构和预训练 + 微调的训练方式相结合，提出一系列 Transformer 模型结构、训练方法的改进

![各种Transformer改进](images/task01/X-formers.png)
