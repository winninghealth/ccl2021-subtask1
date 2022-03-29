## CCL2021 智能对话诊疗评测 命名实体识别

### 任务背景

比赛名称：第一届智能对话诊疗评测比赛（第二十届中国计算语言学大会 CCL2021）

比赛结果：总分第1名（84.72%）

比赛任务：命名实体识别（任务一）第1名（93.22%）

比赛官网：http://www.fudan-disc.com/sharedtask/imcs21/index.html

方案思路：https://zhuanlan.zhihu.com/p/489640773

任务简介：针对互联网医患在线对话的文本，识别出其中5类医疗相关实体（症状、药品名称、药物类别、检查和操作）。该任务采用字符级别的BIO标注体系，因此无实体嵌套的情况。

### 数据集

IMCS21数据集由复旦大学大数据学院在复旦大学医学院专家的指导下构建。本次评测任务使用的IMCS-NER数据集在中文医疗信息处理挑战榜CBLUE持续开放下载，地址：https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

CBLUE公开的3,052条数据包括1,824条训练数据、616条验证数据和612条测试数据。请将下载后的数据保存在`data/dataset`路径下。其中训练和验证数据来自CCL2021评测任务的训练集，测试数据来自CCL2021评测任务的测试集（因此建议将下载后的`IMCS_train.json`和`IMCS_dev.json`合并为`train.json`，与CCL评测任务对齐）。本方案在CCL的评测中对2,440条训练数据进行五折划分，结果可在`data/dataset/5cross_split.csv`中查看（仅供参考），CBLUE已提供官方数据划分文件`split.csv`。

### 环境依赖

- 主要基于Python (3.7.3+)&AllenNLP实现

- 实验使用GPU包括：Tesla V100 和 GeForce GTX 1080Ti

- Python版本依赖：

```
torch==1.7.1+cu101
transformers==4.4.2
allennlp==2.4.0
pypinyin==0.40.0
pandas==1.1.4
numpy==1.19.5
```

### 快速开始

#### 预训练模型

实验中选择了5种不同（规模）的开源预训练模型：

1. MC-BERT，下载地址：https://github.com/alibaba-research/ChineseBLUE
2. RoBERTa-wwm-ext，下载地址：https://huggingface.co/hfl/chinese-roberta-wwm-ext
3. RoBERTa-wwm-ext-large，下载地址：https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
4. PCL-MedBERT，下载地址：https://code.ihub.org.cn/projects/1775
5. medbert-kd-chinese，下载地址：https://huggingface.co/trueto/medbert-kd-chinese

请将下载后的模型权重`pytorch_model.bin`保存在`pretrained_models`路径下相应名称的模型文件夹中

#### 数据预处理

```python
cd data
python data_preprocess.py
```

- 使用CBLUE申请的公开数据集时，建议将`IMCS_train.json`和`IMCS_dev.json`合并为`train.json`，与CCL评测任务对齐
- 按照`data/dataset/5cross_split.csv`中的数据划分进行五折交叉验证，预处理结果保存在`data/ner_data`中

#### 模型训练

```python
python trainer.py --train_file ./data/ner_data/train/pair.seq.diag.char_0 --dev_file ./data/ner_data/dev/pair.seq.diag.char_0 --pretrained_model_dir ./pretrained_models/RoBert_wwm_ext --output_model_dir ./save_model/RoBert_wwm_ext/model_0 --cuda_id cuda:0 --batch_size 16 --num_epochs 20 --patience 5
```

- 参数：{train_file}: 训练数据集路径，{dev_file}: 验证数据集路径，{pretrained_model_dir}: 预训练语言模型路径，{output_model_dir}: 模型保存路径

#### 模型预测

```python
python predict.py --test_input_file ./data/dataset/IMCS_test.json --test_output_file submission.json --model_dir ./save_model/RoBert_wwm_ext/model_0 --pretrained_model_dir ./pretrained_models/RoBert_wwm_ext --cuda_id cuda:0 --batch_size 32 --dialogue_label dialogue_label.json
```

- 参数：{test_input_file}: 测试数据集路径，{test_output_file}: 预测结果输出路径，{model_dir}: 加载的已训练模型路径，{pretrained_model_dir}: 预训练语言模型路径，{dialogue_label}: 整段对话的疾病诊断（diagnosis）标签，由上游任务生成，文件内格式为：{'dialogue_id':'diagnosis_label',...}，该文件暂不提供

#### 其他

- 词向量生成

   本任务使用的词典为：`data/word_vocab/dialogue_dict.txt`，包含2,388个词，`function/saved_word_embedding_-1.pkl`为对应的词向量，通过`word_features.py`生成。具体方法请参考：https://github.com/liuwei1206/LEBERT （Lexicon Enhanced Chinese Sequence Labeling Using BERT Adapter），通过 BERT Adapter 在模型中融入词典信息

- 汉字拆字特征

   参考：https://github.com/howl-anderson/hanzi_chaizi 对汉字字符进行笔画拆分，相关文件：`chaizi.pkl`

### 如何引用

```
@Misc{Jiang2022Shared,
      author={Yiwen Jiang},
      title={CCL2021: First Place Solutions of Named Entity Recognition Task within Online Medical Dialogues},
      year={2022},
      howpublished={GitHub},
      url={https://github.com/winninghealth/ccl2021-subtask1},
}
```

### 版权

MIT License - 详见 [LICENSE](LICENSE)

