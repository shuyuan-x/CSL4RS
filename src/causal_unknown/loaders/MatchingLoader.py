from loaders.BaseLoader import BaseLoader
import os
import pandas as pd
import numpy as np
import pickle
import pdb

class MatchingLoader(BaseLoader):
    def get_hist(self, hist_dict=None):
        """
        get history sequence for data with leave-one-out splitting.
        """
        if hist_dict is None:
            hist_dict = dict()
        uids = self.data['uid'].tolist()
        iids = self.data['iid'].tolist()
        for i, uid in enumerate(uids):
            iid = iids[i]
            if uid not in hist_dict:
                hist_dict[uid] = []
            hist_dict[uid].append(iid)
        self.hist_dict = hist_dict
        
        return
    
    def __len__(self):
        if self.task is not None:
            return len(self.data)
        raise Exception("Set up task first")
        
    def get_neg_eval(self, hist_dict):
        """
        generate negative samples
        """
        if self.task == 'validation':
            neg_num = self.val_neg
        else:
            neg_num = self.test_neg
        
        neg_sample = dict()
        neg_sample['uid'] = []
        neg_sample['iid'] = []
        neg_sample['rating'] = []
        neg_sample['time'] = []
        
        uids = self.data['uid'].tolist()
        iids = self.data['iid'].tolist()
        times = self.data['time'].tolist()
        for i, uid in enumerate(uids):
            iid = iids[i]
            time = times[i]
            if neg_num > 0:
                neg_list = []
                while len(neg_list) < neg_num:
                    neg_id = np.random.randint(self.item_num)
                    if self.pt > 0:
                        if neg_id not in neg_list and neg_id != iid:
                            neg_list.append(neg_id)
                    else:
                        if neg_id not in neg_list and neg_id not in hist_dict[uid]:
                            neg_list.append(neg_id)
            else:
                if self.pt > 0:
                    neg_list = [i for i in range(self.item_num) if i != iid]
                else:
                    neg_list = [i for i in range(self.item_num) if i not in hist_dict[uid]]
            neg_sample['uid'].extend([uid] * len(neg_list))
            neg_sample['iid'].extend(neg_list)
            neg_sample['time'].extend([time] * len(neg_list))
            neg_sample['rating'].extend([0] * len(neg_list))
        self.data = self.data.append(pd.DataFrame(neg_sample))
        return
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {'uid': np.array(row['uid']).astype(np.int64),
                'iid': np.array(row['iid']).astype(np.int64),
                'rating': np.array(row['rating']).astype(np.int64)
               }
    