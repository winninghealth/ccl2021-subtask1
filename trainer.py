# -*- coding: utf-8 -*-
# @author: Yiwen Jiang @Winning Health Group

import os
import torch
import argparse
from typing import Iterable
from allennlp.data import (
    DataLoader,
    Instance,
    Vocabulary,
)
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.training.checkpointer import Checkpointer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder
from allennlp.training.learning_rate_schedulers.linear_with_warmup import LinearWithWarmup

from utils import init_logger
from modeling_tagger import CrfTagger
from word_features import word_preprocess
from transformers.optimization import AdamW
from data_loader import SequenceTaggingDatasetReader

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary, pretrained_word_embedding, transformer_load_path, pretrained_hidden_size) -> Model:
    vocab_size = vocab.get_vocab_size("tokens")
    ngembedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=256, num_embeddings=vocab_size)}
    )
    pyembedder = BasicTextFieldEmbedder(
        {"pinyin": Embedding(embedding_dim=256, num_embeddings=vocab.get_vocab_size("pinyin"))}
    )
    czembedder = BasicTextFieldEmbedder(
        {"chaizi": Embedding(embedding_dim=256, num_embeddings=vocab.get_vocab_size("chaizi"))}
    )
    pyencoder = CnnEncoder(embedding_dim=256,num_filters=128,ngram_filter_sizes=(2,3,4,5),output_dim=256)
    czencoder = CnnEncoder(embedding_dim=256,num_filters=128,ngram_filter_sizes=(3,4,5,6),output_dim=256)
    ngramencoder = FeedForward(input_dim=256,num_layers=2,hidden_dims=256,activations=torch.nn.ReLU(),dropout=0.1)
    lstmencoder = LstmSeq2SeqEncoder(input_size=pretrained_hidden_size+768,hidden_size=768,num_layers=1,dropout=0.1,bidirectional=True)
    return CrfTagger(vocab=vocab,
                     text_field_embedder=ngembedder,
                     pinyin_field_embedder=pyembedder,
                     chaizi_field_embedder=czembedder,
                     pyencoder=pyencoder,
                     czencoder=czencoder,
                     ngramencoder=ngramencoder,
                     lstmencoder=lstmencoder,
                     transformer_load_path=transformer_load_path,
                     pretrained_word_embedding=pretrained_word_embedding,
                     label_encoding='BIO',
                     include_start_end_transitions=False,
                     constrain_crf_decoding=True,
                     calculate_span_f1=True,
                     verbose_metrics=True,
                     ignore_loss_on_o_tags=False)

def build_trainer(model: Model,
                  train_loader: DataLoader,
                  dev_loader: DataLoader,
                  cuda_device: torch.device,
                  serialization_dir: str,
                  num_epochs: int,
                  patience: int
                  ) -> Trainer:
    
    no_bigger = ["text_field_embedder",
                 "pinyin_field_embedder",
                 "chaizi_field_embedder",
                 "pyencoder",
                 "czencoder",
                 "ngramencoder",
                 "lstmencoder",
                 "extra_word_embeddings", 
                 "attn_w",
                 "word_transform", 
                 "word_word_weight", 
                 "tag_projection_layer", 
                 "crf"]
    
    parameter_groups = [
    {
     "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bigger)],
     "weight_decay": 0.0,
    },
    {
     "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bigger)],
     "lr": 0.0001
    }
    ]
    optimizer = AdamW(parameter_groups, lr=1e-5, eps=1e-8)
    lrschedule = LinearWithWarmup(optimizer=optimizer,
                                  num_epochs=num_epochs,
                                  num_steps_per_epoch=len(train_loader),
                                  warmup_steps=190)
    
    ckp = Checkpointer(serialization_dir=serialization_dir,
                       num_serialized_models_to_keep=-1)
    
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     patience=patience,
                                     validation_data_loader=dev_loader,
                                     num_epochs=num_epochs,
                                     serialization_dir=serialization_dir,
                                     cuda_device=cuda_device if str(cuda_device) != 'cpu' else -1,
                                     learning_rate_scheduler=lrschedule,
                                     checkpointer=ckp)
    
    return trainer

def run_training_loop(config):
    
    serialization_dir = config.output_model_dir
    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
    os.makedirs(serialization_dir, exist_ok=True)
    
    word_vocab, lexicon_tree, pretrained_word_embedding, embed_dim = word_preprocess()
    
    dataset_reader = SequenceTaggingDatasetReader(word_vocab,
                                                  lexicon_tree,
                                                  transformer_load_path=config.pretrained_model_dir)
    train_path = config.train_file
    dev_path = config.dev_file
    train_data = list(dataset_reader.read(train_path))
    dev_data = list(dataset_reader.read(dev_path))
    vocab = build_vocab(train_data + dev_data)
    vocab.save_to_files(vocabulary_dir)
    model = build_model(vocab,
                        pretrained_word_embedding,
                        transformer_load_path=config.pretrained_model_dir,
                        pretrained_hidden_size=config.pretrained_hidden_size)
    
    train_loader = MultiProcessDataLoader(dataset_reader, train_path, batch_size=config.batch_size, shuffle=True)
    dev_loader = MultiProcessDataLoader(dataset_reader, dev_path, batch_size=config.batch_size, shuffle=False)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    device = torch.device(config.cuda_id if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    trainer = build_trainer(model,
                            train_loader,
                            dev_loader,
                            device,
                            serialization_dir,
                            config.num_epochs,
                            config.patience)
    trainer.train()
    return trainer

if __name__ == '__main__':
    
    init_logger()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_file", default='./data/ner_data/train/pair.seq.diag.char_0', type=str)
    parser.add_argument("--dev_file", default='./data/ner_data/dev/pair.seq.diag.char_0', type=str)
    parser.add_argument("--output_model_dir", default='./save_model/RoBert_wwm_ext/model_0', type=str)
    parser.add_argument("--pretrained_model_dir", default="./pretrained_models/RoBert_wwm_ext", type=str)
    parser.add_argument("--pretrained_hidden_size", default=768, type=int)
    parser.add_argument("--cuda_id", default='cuda:0', type=str)
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--patience", default=5, type=int)
    
    config = parser.parse_args()
    run_training_loop(config)
    
    