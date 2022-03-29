# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import torch
import logging
from typing import Dict, List
from overrides import overrides
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, ListField, TensorField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer

from utils import HanziChaizi
from pypinyin import lazy_pinyin, Style
from function.preprocess import sent_to_matched_words_boundaries

logger = logging.getLogger(__name__)

DEFAULT_WORD_TAG_DELIMITER = "###"

class SequenceTaggingDatasetReader(DatasetReader):
    def __init__(
        self,
        word_vocab, lexicon_tree,
        transformer_load_path: str,
        max_word_num = 5,
        word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
        token_delimiter: str = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        transformer_indexers: Dict[str, TokenIndexer] = None,
        pinyin_indexers: Dict[str, TokenIndexer] = None,
        chaizi_indexers: Dict[str, TokenIndexer] = None,
        words_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._word_vocab = word_vocab
        self._lexicon_tree = lexicon_tree
        self._max_word_num = max_word_num
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace='tokens')}
        self._transformer_indexers = transformer_indexers or {"transformer": PretrainedTransformerIndexer(model_name=transformer_load_path,namespace='transformer')}
        self._pinyin_indexers = pinyin_indexers or {"pinyin": SingleIdTokenIndexer(namespace='pinyin')}
        self._chaizi_indexers = chaizi_indexers or {"chaizi": SingleIdTokenIndexer(namespace='chaizi')}
        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter

    @overrides
    def _read(self, file_path):
        chaizi_engine = HanziChaizi()
        with open(file_path, "r", encoding='utf-8') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                tokens_and_tags = [
                    pair.rsplit(self._word_tag_delimiter, 1)
                    for pair in line.split(self._token_delimiter)
                ]
                
                tokens = [Token(token) for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]
                
                pinyin = lazy_pinyin(''.join([item.text for item in tokens][1:]), style=Style.TONE3, errors=lambda x:['' for _ in x])
                pinyin = [''] + pinyin
                pinyin = [list(item) for item in pinyin]
                for idx_c, char in enumerate(pinyin):
                    for idx_a, alphabet in enumerate(pinyin[idx_c]):
                        pinyin[idx_c][idx_a] = Token(alphabet)
                assert len(tokens) == len(pinyin)
                
                chaizi = [list(chaizi_engine.form_feature(token)) for token, tag in tokens_and_tags]
                for idx_c, char in enumerate(chaizi):
                    for idx_p, part in enumerate(chaizi[idx_c]):
                        chaizi[idx_c][idx_p] = Token(part)
                assert len(tokens) == len(chaizi)
                
                text = [token.text for token in tokens]
                words = [[0,0,0,0,0]] # '[CLS]'
                matched_words, _ = sent_to_matched_words_boundaries(text, self._lexicon_tree, self._max_word_num)
                for idy in range(len(text)):
                    now_words = matched_words[idy]
                    now_word_ids = self._word_vocab.convert_items_to_ids(now_words)
                    if len(now_word_ids) < self._max_word_num:
                        now_word_ids += [0] * (self._max_word_num - len(now_word_ids))
                    words.append(now_word_ids)
                words.append([0,0,0,0,0]) # '[SEP]'
                
                yield self.text_to_instance(tokens, pinyin, chaizi, words, tags)

    def text_to_instance(
        self,
        tokens: List[Token],
        pinyin: List[List[Token]],
        chaizi: List[List[Token]],
        words: List[List[int]],
        tags: List[str] = None
    ) -> Instance:
        
        fields: Dict[str, Field] = {}
        
        sequence = TextField(tokens, self._token_indexers)
        sequence_bert = TextField([Token('[CLS]')] + tokens + [Token('[SEP]')], self._transformer_indexers)
        sequence_pinyin = [TextField(i, self._pinyin_indexers) for i in pinyin]
        sequence_pinyin = ListField(sequence_pinyin)
        sequence_chaizi = [TextField(i, self._chaizi_indexers) for i in chaizi]
        sequence_chaizi = ListField(sequence_chaizi)
        sequence_words = torch.tensor(words)
        sequence_words = TensorField(sequence_words)
        
        fields["tokens"] = sequence
        fields["pinyin"] = sequence_pinyin
        fields["chaizi"] = sequence_chaizi
        fields["words"] = sequence_words
        fields["pretrained"] = sequence_bert
        
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        
        return Instance(fields)

    