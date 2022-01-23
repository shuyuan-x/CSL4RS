from loaders.MatchingLoader import MatchingLoader
import os
import pandas as pd
import numpy as np
import pickle
import itertools
from collections import defaultdict
import pdb

class CooccurLoader(MatchingLoader):
    def generate_cooccur(self):
        item_pair = defaultdict(int)
        for user in self.dict_hist:
            item_list = self.dict_hist[user]
            for pair in itertools.combinations(item_list, 2):
                item_pair[pair] += 1
        D = len(item_pair)
        item_freq = self.data['iid'].value_counts().to_dict()
        for key in item_pair:
            item_pair[key] = item_pair[key] * D / (item_freq[key[0]] * item_freq[key[1]])
        self.L = D
        self.data = dict()
        self.data['pair'] = list(item_pair.keys())
        self.data['value'] = list(item_pair.values())
            
        return