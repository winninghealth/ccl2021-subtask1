# -*- coding: utf-8 -*-
# ref: https://github.com/liuwei1206/LEBERT

from function.vocab import ItemVocabArray
from function.utils import build_pretrained_embedding_for_corpus
from function.preprocess import build_lexicon_tree_from_vocabs, get_corpus_matched_word_from_lexicon_tree

def word_preprocess(train_data_file = r'./data/dataset/train.json',
                    word_vocab_file = r'./data/word_vocab/dialogue_dict.txt', 
                    embedding_path = r'./data/word_embeds/word_embedding.txt',
                    saved_corpus_embedding_dir='function',
                    max_scan_nums=-1,
                    embed_dim=200):
    
    lexicon_tree = build_lexicon_tree_from_vocabs([word_vocab_file], [max_scan_nums])
    embed_lexicon_tree = lexicon_tree
    matched_words = get_corpus_matched_word_from_lexicon_tree([train_data_file], embed_lexicon_tree)
    word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False, unk_num=5)
    with open('word_vocab.txt', 'w', encoding='utf-8') as f:
        for idx, word in enumerate(word_vocab.idx2item):
            f.write("%d\t%s\n"%(idx, word))
    pretrained_word_embedding, embed_dim = build_pretrained_embedding_for_corpus(
        embedding_path=embedding_path,
        word_vocab=word_vocab,
        embed_dim=embed_dim,
        max_scan_num=max_scan_nums,
        saved_corpus_embedding_dir=saved_corpus_embedding_dir,
    )
    return word_vocab, lexicon_tree, pretrained_word_embedding, embed_dim