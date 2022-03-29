# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.models import Model
from allennlp.data.tokenizers import Token
from allennlp.predictors.predictor import Predictor

from trainer import build_model
from pypinyin import lazy_pinyin, Style
from utils import HanziChaizi, init_logger
from data_loader import SequenceTaggingDatasetReader
from word_features import word_preprocess
from function.preprocess import sent_to_matched_words_boundaries

logger = logging.getLogger(__name__)

special_token2unused = {'小儿消化不良':'[unused1]',
                        '小儿支气管炎':'[unused2]',
                        '上呼吸道感染':'[unused3]',
                        '小儿腹泻':'[unused4]',
                        '小儿发热':'[unused5]',
                        '小儿感冒':'[unused6]'}

class SentenceTaggerPredictor(Predictor):    
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 transformer_load_path: str,
                 word_vocab, lexicon_tree
                 ) -> None:
        super().__init__(model, dataset_reader)
        self.vocab = model.vocab
        self._word_vocab = word_vocab
        self._lexicon_tree = lexicon_tree
        self._max_word_num = 5
        self._token_indexers = {"tokens": SingleIdTokenIndexer(namespace='tokens')}
        self._transformer_indexers = {"transformer": PretrainedTransformerMismatchedIndexer(model_name=transformer_load_path,namespace='transformer')}
        self._pinyin_indexers = {"pinyin": SingleIdTokenIndexer(namespace='pinyin')}
        self._chaizi_indexers = {"chaizi": SingleIdTokenIndexer(namespace='chaizi')}
        self.chaizi_engine = HanziChaizi()
        
    def predict(self, sentence: list) -> JsonDict:
        result = self.predict_batch_json([{"sentence": sent} for sent in sentence])
        instances = [i['tags'] for i in result]
        for idx, i in enumerate(instances):
            instances[idx] = [self.vocab.get_token_from_index(j,namespace='labels') for j in i][4:]
        return instances
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens_wo_tags = sentence
        tokens = [Token(token) for token in tokens_wo_tags]
        
        pinyin = lazy_pinyin(''.join([item.text for item in tokens][1:]), style=Style.TONE3, errors=lambda x:['' for _ in x])
        pinyin = [''] + pinyin
        pinyin = [list(item) for item in pinyin]
        for idx_c, char in enumerate(pinyin):
            for idx_a, alphabet in enumerate(pinyin[idx_c]):
                pinyin[idx_c][idx_a] = Token(alphabet)
        assert len(tokens) == len(pinyin)
        
        chaizi = [list(self.chaizi_engine.form_feature(token)) for token in tokens_wo_tags]
        for idx_c, char in enumerate(chaizi):
            for idx_p, part in enumerate(chaizi[idx_c]):
                chaizi[idx_c][idx_p] = Token(part)
        assert len(tokens) == len(chaizi)
        
        text = [token.text for token in tokens]
        words = [[0,0,0,0,0]]
        matched_words, _ = sent_to_matched_words_boundaries(text, self._lexicon_tree, self._max_word_num)
        for idy in range(len(text)):
            now_words = matched_words[idy]
            now_word_ids = self._word_vocab.convert_items_to_ids(now_words)
            if len(now_word_ids) < self._max_word_num:
                now_word_ids += [0] * (self._max_word_num - len(now_word_ids))
            words.append(now_word_ids)
        words.append([0,0,0,0,0])
        
        return self._dataset_reader.text_to_instance(tokens, pinyin, chaizi, words)

def get_device(pred_config):
    return pred_config.cuda_id if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"

def read_dialogue_label(fn):
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_input_file(input_path, label_dict):
    lines, eids, sids = [], [], []
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    f.close()
    for k, v in data.items():
        for sent in data[k]['dialogue']:
            words = sent['speaker'] + '：' + sent['sentence']
            words = [special_token2unused[label_dict[k]]] + list(words)
            lines.append(words)
            eids.append(k)
            sids.append(sent['sentence_id'])
    return (lines, eids, sids)

def predict(pred_config):
    
    serialization_dir = pred_config.model_dir
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    model_dir = os.path.join(serialization_dir, pred_config.model_name)
    
    vocab = Vocabulary.from_files(vocabulary_dir)
    word_vocab, lexicon_tree, pretrained_word_embedding, embed_dim = word_preprocess()
    
    device = get_device(pred_config)
    model = build_model(vocab, 
                        pretrained_word_embedding, 
                        transformer_load_path=pred_config.pretrained_model_dir, 
                        pretrained_hidden_size=pred_config.pretrained_hidden_size)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model = model.to(device)
    
    dataset_reader = SequenceTaggingDatasetReader(word_vocab,
                                                  lexicon_tree,
                                                  transformer_load_path=pred_config.pretrained_model_dir)
    
    predictor = SentenceTaggerPredictor(model=model,
                                        dataset_reader=dataset_reader,
                                        transformer_load_path=pred_config.pretrained_model_dir,
                                        word_vocab=word_vocab,
                                        lexicon_tree=lexicon_tree)
    
    dialogue_label = read_dialogue_label(pred_config.dialogue_label)    
    (lines, eids, sids) = read_input_file(os.path.join(pred_config.test_input_file), dialogue_label)
    
    batch_size = pred_config.batch_size
    predict_result = dict()
    for i in tqdm(range(0, len(lines), batch_size)):
        for sent_list in [lines[i:i+batch_size]]:
            result = predictor.predict(sent_list)
            for j in zip(eids[i:i+batch_size], sids[i:i+batch_size], result):
                e, s, r = j
                if e not in predict_result:
                    predict_result[e] = dict()
                predict_result[e][s] = ' '.join(r)
    
    pred_path = os.path.join(pred_config.test_output_file)
    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(predict_result, json_file, ensure_ascii=False, indent=4)
    json_file.close()
    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_input_file", default="./data/dataset/test.json", type=str)
    parser.add_argument("--test_output_file", default="submission.json", type=str)
    parser.add_argument("--model_dir", default="./save_model/RoBert_wwm_ext/model_0", type=str)
    parser.add_argument("--model_name", default="best.th", type=str)
    parser.add_argument("--pretrained_model_dir", default="./pretrained_models/RoBert_wwm_ext", type=str)
    
    parser.add_argument("--pretrained_hidden_size", default=768, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--cuda_id", default='cuda:0', type=str)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    # diagnosis label for each whole dialogue predicted by other models
    parser.add_argument("--dialogue_label", default='dialogue_label.json', type=str)
    
    pred_config = parser.parse_args()
    predict(pred_config)
    
    