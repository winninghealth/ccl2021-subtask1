# -*- coding: utf-8 -*-
# some code ref: https://github.com/howl-anderson/hanzi_chaizi
# Modified by @author: Yiwen Jiang @Winning Health Group

import copy
import pickle
import logging
import pkg_resources

class HanziChaizi(object):
    def __init__(self):
        
        data_file = pkg_resources.resource_filename(__name__, "./data/chaizi.pkl")
        with open(data_file, 'rb') as fd:
            self.data = pickle.load(fd)
        fd.close()
        self.special_token = '離'
    
    def query(self, input_char, default=None):
        result = self.data.get(input_char, default)
        return result
    
    def _flatten(self, li):
        return sum(([x] if not isinstance(x, list) else self._flatten(x) for x in li), [])
    
    def form_feature(self, input_char):
        feature = []
        input_list = [input_char]
        continue_chai = True
        while continue_chai == True:
            input_list_tmp = copy.deepcopy(input_list)
            for idx, char in enumerate(input_list):
                char_query = self.query(char)
                if char_query != None:
                    char_query_tmp = char_query[0]
                    if char in char_query_tmp and len(char_query) > 1:
                        for result in char_query:
                            if char not in result:
                                char_query_tmp = result
                                break
                    if char in char_query_tmp:
                        while char in char_query_tmp:
                            char_query_tmp.remove(char)
                    if self.special_token in char_query_tmp:
                        while self.special_token in char_query_tmp:
                            char_query_tmp.remove(self.special_token)
                    char_query = char_query_tmp
                else:
                    char_query = [char]
                input_list_tmp[idx] = char_query
            input_list_tmp = self._flatten(input_list_tmp)
            if input_list != input_list_tmp:
                continue_chai = True
                feature += input_list_tmp
                input_list = copy.deepcopy(input_list_tmp)
                if len(feature) > 200:
                    feature = feature[:200]
                    break
            else:
                continue_chai = False
        return feature[:80]


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

