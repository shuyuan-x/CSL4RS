from loaders.HistLoader import HistLoader
import os
import pandas as pd
import numpy as np
import pickle
import pdb

class CausalDiscoveryLoader(HistLoader):
    
    def parse_loader_args(parser):
        parser.add_argument('--intervention_ratio', default=0.5, type=float,
                           help='the ratio of interventional data in training data.')
        parser.add_argument('--train_neg', default=1, type=float,
                           help='the number of negative samples during the training.')
        return HistLoader.parse_loader_args(parser)
    
    def __init__(self, intervention_ratio, train_neg, *args, **kwargs):
        HistLoader.__init__(self, *args, **kwargs)
        self.intervention_ratio = intervention_ratio
        self.train_neg = train_neg
        self.intervention_data = None
        self.observation_data = None
#         self.item_num -= 1 # no 0 as dummy history
        
    def _getitem_train(self, idx):
        row = self.data.iloc[idx]
        return {'uid': np.array(row['uid']).astype(np.int64),
                'iid': np.array(row['iid']).astype(np.int64),
                'negative': np.array([int(i) for i in row['history'].split(',')]).astype(np.int64),
                'history': np.array([int(i) for i in row['history'].split(',')]).astype(np.int64)}
    
    def _get_neg_train_df(self, hist_dict):
        uids = self.data['uid'].tolist()
        iids = self.data['iid'].tolist()
        negs = []
        for i, uid in enumerate(uids):
            neg_list = []
            iid = iids[i]
            while len(neg_list) < self.train_neg:
                neg_id = np.random.randint(1, self.item_num)
                if self.pt > 0:
                    if neg_id not in neg_list and neg_id != iid:
                        neg_list.append(neg_id)
                else:
                    if neg_id not in neg_list and neg_id not in hist_dict[uid]:
                        neg_list.append(neg_id)
            negs.append(str(neg_list).replace(' ', '')[1:-1])
        self.data['negative'] = negs
        self.L = len(self.data)
        return
    
    def get_data(self, intervention=False):
        if self.intervention_data is None:
            self._split_intervention()
        if intervention:
            self.data = self.intervention_data
            self.L = len(self.intervention_data)
        else:
            self.data = self.observation_data
            self.L = len(self.observation_data)
        return
        
    def _split_intervention(self):
        """
        split interventional data from training data.
        """
        if self.intervention_data is not None:
            return
        if self.task == 'train':
            split_index = []
            for uid, group in self.data.groupby('uid'):
                num = int(len(group) * self.intervention_ratio)
                split_index.extend(group.index[-num:])
            self.intervention_data = self.data.loc[split_index].reset_index()
            self.observation_data = self.data.drop(split_index).reset_index()
            
        return
        