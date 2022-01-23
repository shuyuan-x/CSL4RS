from loaders.CausalDiscoveryLoader import CausalDiscoveryLoader
import os
import pandas as pd
import numpy as np
import pickle
import pdb

class GRUCausalDiscoveryLoader(CausalDiscoveryLoader):
    
    def _get_neg_train_df(self, hist_dict):
        uids = self.data['uid'].tolist()
        iids = self.data['iid'].tolist()
        negs = []
        for i, uid in enumerate(uids):
            while True:
                neg_id = np.random.randint(self.item_num)
                if neg_id not in hist_dict[uid]:
                    negs.append(neg_id)
                    break
        self.data['negative'] = negs
        self.L = len(self.data)
        return
    
    def _getitem_train(self, idx):
        row = self.data.iloc[idx]
        return {'uid': np.array(row['uid']).astype(np.int64),
                'iid': np.array(row['iid']).astype(np.int64),
                'negative': np.array(row['negative']).astype(np.int64),
                'history': np.array([int(i) for i in row['history'].split(',')]).astype(np.int64)
#                 'history': row['history']
               }