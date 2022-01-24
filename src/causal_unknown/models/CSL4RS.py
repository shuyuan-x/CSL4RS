from models.BaseModel import BaseModel
from utils.dag import GumbelAdjacency
import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb

class CSL4RS(BaseModel):
    loader = 'HistLoader'
    runner = 'CSL4RSRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--hidden_size_MLP', default=4, type=int,
                           help='Size of hidden vectors')
        parser.add_argument('--hidden_size_GRU', default=64, type=int,
                           help='Size of hidden vectors in GRU')
        parser.add_argument('--num_layers', default=1, type=int,
                           help='Number of GRU layers')
        parser.add_argument('--emb_size', default=64, type=int,
                           help='Size of embedding')
        return BaseModel.parse_model_args(parser)
    
    def __init__(self, hidden_size_MLP, hidden_size_GRU, num_layers, emb_size, *args, **kwargs):
        self.hidden_size_MLP = hidden_size_MLP
        self.hidden_size_GRU = hidden_size_GRU
        self.num_layers = num_layers
        self.emb_size = emb_size
        BaseModel.__init__(self, *args, **kwargs)
        
    def _init_weight(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.rnn = torch.nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size_GRU, batch_first=True, num_layers=self.num_layers)
        self.out = torch.nn.Linear(self.hidden_size_GRU, self.emb_size, bias=False)
        self.cat_num = 1
        Ns = self.item_num * self.cat_num
        M = self.item_num
        H = self.hidden_size_MLP
        self.W0 = torch.nn.Parameter(torch.empty(M, Ns, H))
        self.B0 = torch.nn.Parameter(torch.empty(M, H))
        self.W1 = torch.nn.Parameter(torch.empty(M, H, self.cat_num))
        self.B1 = torch.nn.Parameter(torch.empty(M, self.cat_num))
        torch.nn.init.uniform_(self.W0, b=0.1)
        torch.nn.init.uniform_(self.W1, b=0.1)
        torch.nn.init.uniform_(self.B0, b=0.1)
        torch.nn.init.uniform_(self.B1, b=0.1)
        self.gumbel_adj = GumbelAdjacency(self.item_num)
        self.LeakyRelu = torch.nn.LeakyReLU(0.1)
        self.adj = torch.ones((self.item_num, self.item_num)) - torch.eye(self.item_num)
        
        
    def predict(self, batch):
        uids = batch['uid'].to(torch.long).to(self.W0.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.W0.device)
        hist = batch['history'].to(torch.long).to(self.W0.device)
        label = batch['rating'].to(torch.long).to(self.W0.device).view([-1])
        config = self.gumbel_adj(iids)
        mask = config.scatter_(1, iids.unsqueeze(1), 0)
        mask[:,0] = 0
        
        # MLP
        one_hot = F.one_hot(hist, num_classes=self.item_num).sum(dim=1).gt_(torch.tensor(0).to(self.W0.device))
        
        masked_input = mask.to(torch.float) * one_hot.to(torch.float)
        format_input = masked_input.view(masked_input.shape[0], 1, masked_input.shape[1])
        hidden_pred = torch.matmul(format_input, self.W0[iids]).reshape(masked_input.shape[0],self.hidden_size_MLP) + self.B0[iids]
        hidden_input = self.LeakyRelu(hidden_pred).view(hidden_pred.shape[0], 1, hidden_pred.shape[1])
        pred = torch.matmul(hidden_input, self.W1[iids]).reshape(masked_input.shape[0],self.cat_num) + self.B1[iids]
        mlp_prediction = pred.sigmoid().view([-1])
        
        
        out_dict = {'prediction': mlp_prediction, 'uid': uids, 'label': label}
        
        return out_dict
        
    def estimate(self, batch):
        uids = batch['uid'].to(torch.long).to(self.W0.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.W0.device)
        hist = batch['history'].to(torch.long).to(self.W0.device)
#         label = batch['rating'].to(torch.long).to(self.W0.device).view([-1])
        config = self.gumbel_adj(iids)
        
#         pdb.set_trace()
        mask = config.clone().scatter_(1, iids.unsqueeze(1), 0)
        mask[:,0] = 0
        
        hist_one_hot = F.one_hot(hist, num_classes=self.item_num).sum(dim=1)
        R = torch.cumprod(torch.pow((1-config), hist_one_hot), dim=1)[:,-1]
        R = torch.empty_like(R).uniform_().lt_(R)
        
#         # MLP
        one_hot = hist_one_hot.clone().gt_(torch.tensor(0).to(self.W0.device))
        
        
        masked_input = mask.to(torch.float) * one_hot.to(torch.float)
        format_input = masked_input.view(masked_input.shape[0], 1, masked_input.shape[1])
        hidden_pred = torch.matmul(format_input, self.W0[iids]).reshape(masked_input.shape[0],self.hidden_size_MLP) + self.B0[iids]
        hidden_input = self.LeakyRelu(hidden_pred).view(hidden_pred.shape[0], 1, hidden_pred.shape[1])
        pred = torch.matmul(hidden_input, self.W1[iids]).reshape(masked_input.shape[0],self.cat_num) + self.B1[iids]
        mlp_prediction = pred.sigmoid().view([-1])
        
        # GRU
        valid_his = hist.gt(0).long()
        his_length = valid_his.sum(dim=-1)
        
        his_vectors = self.iid_embeddings(hist) * valid_his.unsqueeze(dim=-1).float()
        
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=hist.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        
        # pack
        packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        
        sorted_rnn_vector = self.out(hidden[-1])
        
        # unsort
        unsorted_idx = torch.topk(sorted_idx, k=hist.shape[0], largest=False)[1]
        rnn_vec = sorted_rnn_vector.index_select(dim=0, index=unsorted_idx)
        
        # predict
        item_vec = self.iid_embeddings(iids)
        gru_prediction = (rnn_vec * item_vec).sum(dim=1).view([-1]).sigmoid()
        
#         prediction = gru_prediction
        prediction = torch.pow(mlp_prediction, 1-R) * torch.pow(gru_prediction, R)
        
        out_dict = {'prediction': prediction}
        return out_dict
    
    def forward(self, batch):
        out_dict = self.estimate(batch)
        
        out_dict['loss'] = -out_dict['prediction'].log().mean()
        return out_dict
    
    def get_w_adj(self):
        """Get adjacency matrix"""
#         pdb.set_trace()
        self.adj = self.adj.to(self.W0.device)
        return self.gumbel_adj.get_prob() * self.adj
    
    def l2(self):
        l2 = 0
        for name, p in self.named_parameters():
            if name not in ['gumbel_adj.log_alpha']:
                l2 += p.norm(2)
        return l2