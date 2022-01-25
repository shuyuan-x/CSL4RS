from models.BaseModel import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb

class item2vec(BaseModel):
    loader = 'HistLoader'
    runner = 'BaseRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--emb_size', default=64, type=int,
                           help='Size of embedding')
        return BaseModel.parse_model_args(parser)
    
    def __init__(self, emb_size, *args, **kwargs):
        self.emb_size = emb_size
        BaseModel.__init__(self, *args, **kwargs)
        
    def _init_weight(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.emb_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.loss_func = torch.nn.MSELoss()
        return
    
    def predict(self, batch):
        """
        prediction for evaluation
        """
        
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        label = batch['rating'].to(torch.long).to(self.dummy_param.device).view([-1])

        # predict
        user_vec = self.uid_embeddings(uids)
        item_vec = self.iid_embeddings(iids)
        prediction = (user_vec * item_vec).sum(dim=1).view([-1])
        
        out_dict = {'prediction': prediction, 'uid': uids, 'label': label}
        return out_dict
        
    def estimate(self, batch):
        """
        estimation for training
        """
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        negs = batch['negative'].to(torch.long).to(self.dummy_param.device).view([-1])
        pad_his = batch['history'].to(torch.long).to(self.dummy_param.device)
        
        bs, his_lens = pad_his.shape[0], pad_his.shape[1]
        
        iids = iids.view(-1,1).expand(bs, his_lens).reshape(-1)
        negs = negs.view(-1,1).expand(bs, his_lens).reshape(-1)
        his = pad_his.reshape(-1)
        
        his_item_vec = self.iid_embeddings(his)
        
        pos_item_vec = self.iid_embeddings(iids)
        pos_prediction = (his_item_vec * pos_item_vec).sum(dim=1).view([-1])
        neg_item_vec = self.iid_embeddings(negs)
        neg_prediction = (his_item_vec * neg_item_vec).sum(dim=1).view([-1])
        out_dict = {'pos_prediction': pos_prediction, 'neg_prediction': neg_prediction}
        
        
        return out_dict
    
    def forward(self, batch):
        """
        calculate the loss
        """
        out_dict = self.estimate(batch)
        pos, neg = out_dict['pos_prediction'], out_dict['neg_prediction']
        loss = -(pos - neg).sigmoid().log().sum()
        out_dict['loss'] = loss
        return out_dict