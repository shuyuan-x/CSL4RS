from models.BaseModel import BaseModel
from utils.dag import GumbelAdjacency
import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb

class notears(BaseModel):
    loader = 'HistLoader'
    runner = 'CSL4RSRunner'
    
        
    def _init_weight(self):
        self.gumbel_adj = GumbelAdjacency(self.item_num)
        self.adj = torch.ones((self.item_num, self.item_num)) - torch.eye(self.item_num)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.loss = torch.nn.MSELoss()
        
    def predict(self, batch):
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        hist = batch['history'].to(torch.long).to(self.dummy_param.device)
        label = batch['rating'].to(torch.long).to(self.dummy_param.device).view([-1])
        config = self.gumbel_adj.get_prob()[iids]
        one_hot = F.one_hot(hist, num_classes=self.item_num).sum(dim=1).gt_(torch.tensor(0).to(self.dummy_param.device))
        
        prediction = (config * one_hot).sum(dim=1).sigmoid().view([-1])
        
        out_dict = {'prediction': prediction, 'uid': uids, 'label': label}
        
        return out_dict
        
    def estimate(self, batch):
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        hist = batch['history'].to(torch.long).to(self.dummy_param.device)
        config = self.gumbel_adj.get_prob()[iids]
        one_hot = F.one_hot(hist, num_classes=self.item_num).sum(dim=1).gt_(torch.tensor(0).to(self.dummy_param.device))
        
        prediction = (config * one_hot).sum(dim=1).sigmoid().view([-1])
        
        out_dict = {'prediction': prediction}
        return out_dict
    
    def forward(self, batch):
        iids = batch['iid'].to(torch.float).to(self.dummy_param.device).view([-1])
        out_dict = self.estimate(batch)
        target = torch.ones_like(iids)
        
        out_dict['loss'] = self.loss(out_dict['prediction'], target)
        return out_dict
    
    def get_w_adj(self):
        """Get adjacency matrix"""
#         pdb.set_trace()
        self.adj = self.adj.to(self.dummy_param.device)
        return self.gumbel_adj.get_prob() * self.adj