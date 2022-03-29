# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group
# Modified from https://github.com/allenai/allennlp-models/blob/main/allennlp_models/tagging/models/crf_tagger.py

import torch
import allennlp.nn.util as util

from overrides import overrides
from torch.nn.modules.linear import Linear
from typing import Dict, Optional, List, cast

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, Seq2VecEncoder, ConditionalRandomField, FeedForward

from bert_adapter import BertAdapterModel

class CrfTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        pinyin_field_embedder: TextFieldEmbedder,
        chaizi_field_embedder: TextFieldEmbedder,
        pyencoder: Seq2VecEncoder,
        czencoder: Seq2VecEncoder,
        ngramencoder: FeedForward,
        lstmencoder: Seq2SeqEncoder,
        transformer_load_path,
        pretrained_word_embedding,
        label_namespace: str = "labels",
        label_encoding: Optional[str] = None,
        include_start_end_transitions: bool = False,
        constrain_crf_decoding: bool = None,
        calculate_span_f1: bool = True,
        dropout: Optional[float] = None,
        verbose_metrics: bool = False,
        top_k: int = 1,
        ignore_loss_on_o_tags: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.text_field_embedder = text_field_embedder
        self.pinyin_field_embedder = pinyin_field_embedder
        self.chaizi_field_embedder = chaizi_field_embedder
        self.pyencoder = pyencoder
        self.czencoder = czencoder
        self.ngramencoder = ngramencoder
        self.lstmencoder = lstmencoder
        self.pretrained_encoder = BertAdapterModel.from_pretrained(transformer_load_path, pretrained_embeddings=pretrained_word_embedding)
        self.label_namespace = label_namespace
        self.label_encoding = label_encoding
        self.include_start_end_transitions = include_start_end_transitions
        self.calculate_span_f1 = calculate_span_f1
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._verbose_metrics = verbose_metrics
        self.top_k = top_k
        self.ignore_loss_on_o_tags = ignore_loss_on_o_tags
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.tag_projection_layer = TimeDistributed(Linear(self.lstmencoder.get_output_dim(), self.num_tags))
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None
        self.crf = ConditionalRandomField(
            self.num_tags, constraints, include_start_end_transitions=include_start_end_transitions
        )
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError(
                    "calculate_span_f1 is True, but no label_encoding was specified."
                )
            self._f1_metric = SpanBasedF1Measure(
                vocab, tag_namespace=label_namespace, label_encoding=label_encoding
            )
        initializer(self)
    
    @overrides
    def forward(
        self,
        tokens: TextFieldTensors,
        pinyin: TextFieldTensors,
        chaizi: TextFieldTensors,
        words: torch.Tensor,
        pretrained: TextFieldTensors,
        tags: torch.LongTensor = None,
        ignore_loss_on_o_tags: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        ignore_loss_on_o_tags = (
            ignore_loss_on_o_tags
            if ignore_loss_on_o_tags is not None
            else self.ignore_loss_on_o_tags
        )
        
        '''
        pinyin feature
        '''
        embedded_pinyin_input = self.pinyin_field_embedder(pinyin)
        pinyin_shape = embedded_pinyin_input.shape
        embedded_pinyin_input = embedded_pinyin_input.view(pinyin_shape[0] * pinyin_shape[1], pinyin_shape[2], pinyin_shape[3])
        mask_pinyin = util.get_text_field_mask(pinyin,num_wrapping_dims=1)
        mask_shape = mask_pinyin.shape
        mask_pinyin = mask_pinyin.view(mask_shape[0] * mask_shape[1], -1)
        if self.dropout:
            embedded_pinyin_input = self.dropout(embedded_pinyin_input)
        encoded_pinyin = self.pyencoder(embedded_pinyin_input, mask_pinyin)
        if self.dropout:
            encoded_pinyin = self.dropout(encoded_pinyin)
        
        '''
        chaizi feature
        '''
        embedded_chaizi_input = self.chaizi_field_embedder(chaizi)
        chaizi_shape = embedded_chaizi_input.shape
        embedded_chaizi_input = embedded_chaizi_input.view(chaizi_shape[0] * chaizi_shape[1], chaizi_shape[2], chaizi_shape[3])
        mask_chaizi = util.get_text_field_mask(chaizi,num_wrapping_dims=1)
        mask_shape = mask_chaizi.shape
        mask_chaizi = mask_chaizi.view(mask_shape[0] * mask_shape[1], -1)
        if self.dropout:
            embedded_chaizi_input = self.dropout(embedded_chaizi_input)
        encoded_chaizi = self.czencoder(embedded_chaizi_input, mask_chaizi)
        if self.dropout:
            encoded_chaizi = self.dropout(encoded_chaizi)
        
        '''
        ngram feature
        '''
        embedded_ngram_input = self.text_field_embedder(tokens)
        if self.dropout:
            embedded_ngram_input = self.dropout(embedded_ngram_input)
        encoded_ngram = self.ngramencoder(embedded_ngram_input)
        
        '''
        pretrained feature
        '''
        input_ids = pretrained['transformer']['token_ids']
        attention_mask = pretrained['transformer']['mask']
        token_type_ids = pretrained['transformer']['type_ids']
        encoded_pretrained = self.pretrained_encoder(input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     token_type_ids=token_type_ids,
                                                     matched_word_ids=words,
                                                     matched_word_mask=(words!=0))
        encoded_pretrained = encoded_pretrained[:,1:-1,:]
        
        '''
        multi-feature concat
        '''
        encoded_chaizi = encoded_chaizi.view(encoded_ngram.shape[0],encoded_ngram.shape[1],encoded_ngram.shape[2])
        encoded_pinyin = encoded_pinyin.view(encoded_ngram.shape[0],encoded_ngram.shape[1],encoded_ngram.shape[2])
        embedded_text_input = torch.cat((encoded_ngram, encoded_chaizi),2)
        embedded_text_input = torch.cat((embedded_text_input, encoded_pinyin),2)
        embedded_text_input = torch.cat((embedded_text_input, encoded_pretrained),2)
        
        '''
        encoder
        '''
        mask = util.get_text_field_mask(tokens)
        encoded_text = self.lstmencoder(embedded_text_input, mask)
        
        '''
        decoder
        '''
        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits,mask,top_k=self.top_k)
        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])
        output = {"logits": logits, "mask": mask, "tags": predicted_tags}
        
        '''
        loss and metric
        '''
        if tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                crf_mask = mask & (tags != o_tag_index)
            else:
                crf_mask = mask
            log_likelihood = self.crf(logits, tags, crf_mask)
            output["loss"] = -log_likelihood
            
            # Represent viterbi tags as "class probabilities" that we can feed into the metrics
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            
            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask)
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, tags, mask)
        
        return output
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }
        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return

