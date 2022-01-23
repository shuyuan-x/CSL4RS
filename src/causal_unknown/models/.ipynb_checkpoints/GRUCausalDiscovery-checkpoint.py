from models.GRU4Rec import GRU4Rec
import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb

class GRUCausalDiscovery(GRU4Rec):
    loader = 'GRUCausalDiscoveryLoader'
    runner = 'CausalDiscoveryRunner'
    
    
    def _init_weight(self):
        self.gamma = torch.nn.Parameter(torch.empty(self.item_num, self.item_num))
        with torch.no_grad(): self.gamma.diagonal().fill_(float("-inf"))
        self.LeakyRelu = torch.nn.LeakyReLU(0.1)
        GRU4Rec._init_weight(self)
        
    def parameters(self):
        s = set(self.structural_parameters())
        l = [p for p in super().parameters() if p not in s]
        return iter(l)
    
    def structural_parameters(self):
        return iter([self.gamma])
    
    def gl2(self):
        l2 = 0
        for p in self.structural_parameters():
            l2 += (p ** 2).sum()
            
        return l2
    
    def predict(self, batch):
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        pad_his = batch['history'].to(torch.long).to(self.dummy_param.device)
        label = batch['rating'].to(torch.long).to(self.dummy_param.device).view([-1])
#         pdb.set_trace()
        with torch.no_grad():
            gammaexp = self.gamma.sigmoid().gt_(torch.tensor(0.5).to(self.dummy_param.device))
#             self.gamma.sigmoid()
            gammaexp.diagonal().zero_()
        
        links = gammaexp[iids]
        links[:,0] = 0
#         hist = [torch.tensor([int(i) for i in row.split(',')]) for row in batch['history']]
#         pad_his = torch.nn.utils.rnn.pad_sequence(hist, batch_first=True).to(self.dummy_param.device)
        
        # mask input based on adjacency matrix
        mask = torch.gather(links, 1, pad_his)
        max_his = pad_his.shape[1]
        new_mask = mask * torch.tensor([max_his - i for i in range(max_his)]).to(self.dummy_param.device)
        value, indice = torch.sort(new_mask, dim=-1, descending=True)
        masked_pad_his = torch.gather(pad_his*mask, 1, indice)
        
        valid_his = masked_pad_his.gt(0).long()
        his_length = valid_his.sum(dim=-1)
        
        try:
            his_vectors = self.iid_embeddings(masked_pad_his.long()) * valid_his.unsqueeze(dim=-1).float()
        except:
            pdb.set_trace()
        
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=pad_his.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        sorted_his_length[sorted_his_length == 0] += 1
        
        # pack
        try:
            packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        except:
            pdb.set_trace()
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        
        sorted_rnn_vector = self.out(hidden[-1])
        
        # unsort
        unsorted_idx = torch.topk(sorted_idx, k=pad_his.shape[0], largest=False)[1]
        rnn_vec = sorted_rnn_vector.index_select(dim=0, index=unsorted_idx)
        
        # predict
        item_vec = self.iid_embeddings(iids)
        prediction = (rnn_vec * item_vec).sum(dim=1).view([-1])
        
        
        out_dict = {'prediction': prediction, 'uid': uids, 'label': label}
        return out_dict
    
    def estimate(self, batch, config):
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        pad_his = batch['history'].to(torch.long).to(self.dummy_param.device)
        
        links = config[iids]
        links[:,0] = 0
#         hist = [torch.tensor([int(i) for i in row.split(',')]) for row in batch['history']]
#         pad_his = torch.nn.utils.rnn.pad_sequence(hist, batch_first=True).to(self.dummy_param.device)
        
        # mask input based on adjacency matrix
        mask = torch.gather(links, 1, pad_his)
        max_his = pad_his.shape[1]
        new_mask = mask * torch.tensor([max_his - i for i in range(max_his)]).to(self.dummy_param.device)
        value, indice = torch.sort(new_mask, dim=-1, descending=True)
        masked_pad_his = torch.gather(pad_his*mask, 1, indice)
        
        valid_his = masked_pad_his.gt(0).long()
        his_length = valid_his.sum(dim=-1)
        
        try:
            his_vectors = self.iid_embeddings(masked_pad_his.long()) * valid_his.unsqueeze(dim=-1).float()
        except:
            pdb.set_trace()
        
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=pad_his.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        sorted_his_length[sorted_his_length == 0] += 1
        
        # pack
        try:
            packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        except:
            pdb.set_trace()
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        
        sorted_rnn_vector = self.out(hidden[-1])
        
        # unsort
        unsorted_idx = torch.topk(sorted_idx, k=pad_his.shape[0], largest=False)[1]
        rnn_vec = sorted_rnn_vector.index_select(dim=0, index=unsorted_idx)
        
        # predict
        item_vec = self.iid_embeddings(iids)
        prediction = (rnn_vec * item_vec).sum(dim=1).view([-1])
        
        format_pred = torch.zeros_like(links).type(torch.float)
        format_pred[:,iids]=prediction
        
        out_dict = {'prediction': prediction, 'format_pred': format_pred}
        return out_dict
    
    def forward(self, batch, intervention=False):
        with torch.no_grad():
            config = self.gamma.sigmoid()
            config = torch.empty_like(config).uniform_().lt_(config)
            config.diagonal().zero_()
        
        out_dict = self.estimate(batch, config)
#         pdb.set_trace()
        
        if intervention:
            out_dict['format_pred'] = out_dict['format_pred'].sum(0) / (out_dict['format_pred']!=0).sum(0)
            out_dict['format_pred'][torch.isnan(out_dict['format_pred'])] = 0
            node = torch.argmax(-out_dict['format_pred']).item()
            out_dict['format_pred'][node]=0#out_dict['format_pred'][node]/2
            
        out_dict['loss'] = -out_dict['format_pred'].mean()
        out_dict['config'] = config
        return out_dict
    
    def reconstrain_gamma(self):
        with torch.no_grad():
            self.gamma.clamp_(-5,+5)
            self.gamma.diagonal().fill_(float("-inf"))