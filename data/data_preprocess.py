# -*- coding:utf-8 -*-
# ref: https://github.com/lemuria-wchen/imcs21-cblue/blob/main/task1/BERT-NER/data/data_preprocess.py
# was finally modified by Yiwen Jiang @Winning Health Group

import pandas as pd
import json
import os

"""显式嵌入对话的diagnosis标签"""
special_token2unused = {'小儿消化不良':'[unused1]',
                        '小儿支气管炎':'[unused2]',
                        '上呼吸道感染':'[unused3]',
                        '小儿腹泻':'[unused4]',
                        '小儿发热':'[unused5]',
                        '小儿感冒':'[unused6]'}

def read_train_data(fn):
    """读取用于训练的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_test_data(fn):
    """读取用于测试的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_example_ids(fn):
    """读取划分数据集的文件"""
    example_ids = pd.read_csv(fn)
    return example_ids

def save_train_data(data, example_ids, idx, mode, fn1, fn2):
    """
    训练集和验证集的数据转换
    :param data: 用于训练的json数据
    :param example_ids: 样本id划分数据
    :idx: index of fold in Cross-validation
    :param mode: train/dev
    :param fn1: BIO序列标签
    :param fn2: BIO序列标签 with diagnosis label
    :return:
    """
    
    eids = example_ids[example_ids['split_' + str(idx)] == mode]['example_id'].to_list()
    seq_in, seq_bio = [], []
    seq_in_diag, seq_bio_diag = [], []
    for eid in eids:
        tmp_data = data[str(eid)]
        tmp_dialogue = tmp_data['dialogue']
        for i in range(len(tmp_dialogue)):
            tmp_sent = list(tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence'])
            tmp_sent_diag = [special_token2unused[tmp_data['diagnosis']]] + list(tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence'])
            tmp_bio = ['O'] * 3 + tmp_dialogue[i]['BIO_label'].split(' ')
            tmp_bio_diag = ['O'] * 4 + tmp_dialogue[i]['BIO_label'].split(' ')
            assert len(tmp_sent) == len(tmp_bio)
            assert len(tmp_sent_diag) == len(tmp_bio_diag)
            seq_in.append(tmp_sent)
            seq_bio.append(tmp_bio)
            seq_in_diag.append(tmp_sent_diag)
            seq_bio_diag.append(tmp_bio_diag)
    assert len(seq_in) == len(seq_bio)
    assert len(seq_in_diag) == len(seq_bio_diag)
    print(mode, '句子数量为：', len(seq_in))
    # 数据保存
    with open(fn1, 'w', encoding='utf-8') as f1:
        for i in range(0, len(seq_in)):
            for j in range(0, len(seq_in[i])):
                f1.write(seq_in[i][j])
                f1.write('###')
                f1.write(seq_bio[i][j])
                f1.write(' \t ')
            f1.write('\n')
    f1.close()
    with open(fn2, 'w', encoding='utf-8') as f2:
        for i in range(0, len(seq_in_diag)):
            for j in range(0, len(seq_in_diag[i])):
                f2.write(seq_in_diag[i][j])
                f2.write('###')
                f2.write(seq_bio_diag[i][j])
                f2.write(' \t ')
            f2.write('\n')
    f2.close()

if __name__ == "__main__":
    
    train_data = read_train_data('./dataset/train.json')
    example_ids = read_example_ids('./dataset/5cross_split.csv')
    
    data_dir = 'ner_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir+'/train')
        os.makedirs(data_dir+'/dev')
    
    fold_num = 5
    for idx in range(fold_num):
        # 获得训练数据
        save_train_data(
            train_data,
            example_ids,
            idx,
            'train',
            os.path.join(data_dir, 'train', 'pair.seq.char_'+str(idx)),
            os.path.join(data_dir, 'train', 'pair.seq.diag.char_'+str(idx))
        )
        
        # 获得验证数据
        save_train_data(
            train_data,
            example_ids,
            idx,
            'dev',
            os.path.join(data_dir, 'dev', 'pair.seq.char_'+str(idx)),
            os.path.join(data_dir, 'dev', 'pair.seq.diag.char_'+str(idx))
        )

