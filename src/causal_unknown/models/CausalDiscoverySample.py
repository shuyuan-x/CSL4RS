from models.CausalDiscovery import CausalDiscovery
import torch
import torch.nn.functional as F
import numpy as np
import pdb


class CausalDiscoverySample(CausalDiscovery):
    loader = 'CausalDiscoveryLoader'
    runner = 'CausalDiscoveryRunner'
    """
    Causal Discovery model with sampled negative items.
    """
    
    def __init__(self, *args, **kwargs):
#         self.hidden_size = hidden_size
        CausalDiscovery.__init__(self, *args, **kwargs)
        
            
    def estimate(self, batch, config):
        uids = batch['uid'].to(torch.long).to(self.W0.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.W0.device)
        hist = batch['history'].to(torch.long).to(self.W0.device)
        negs = batch['negative'].to(torch.long).to(self.W0.device)
        
        
        samples = torch.cat([hist, iids.reshape(iids.shape[0], 1)], dim=1)
        one_hot_1 = F.one_hot(samples, num_classes=self.item_num).sum(dim=1).gt_(torch.tensor(0).to(self.W0.device))
        one_hot_0 = torch.ones_like(one_hot_1) - one_hot_1
        one_hot = torch.cat([one_hot_0.t(), one_hot_1.t()], dim=-1).view(-1, one_hot_0.shape[0]).t() 
        
#         samples_include_neg = torch.cat([samples, negs.reshape(negs.shape[0], 1)], dim=1)
        samples_include_neg = torch.cat([samples, negs], dim=1)
        
        cal_items = samples_include_neg.reshape(-1)
        
        mask = torch.repeat_interleave(config[cal_items], self.cat_num, dim=1)
        
        one_hot = one_hot.view(one_hot.shape[0],1,one_hot.shape[1]).expand(one_hot.shape[0],samples_include_neg.shape[1], one_hot.shape[1])
        one_hot = one_hot.reshape(-1, one_hot.shape[2])
        masked_input = mask.to(torch.float) * one_hot.to(torch.float)
        
        
        format_input = masked_input.view(masked_input.shape[0], 1, masked_input.shape[1])
        hidden_pred = torch.matmul(format_input, self.W0[cal_items]).reshape(masked_input.shape[0],self.hidden_size) + self.B0[cal_items]
        hidden_input = self.LeakyRelu(hidden_pred).view(hidden_pred.shape[0], 1, hidden_pred.shape[1])
#         pdb.set_trace()
        pred = torch.matmul(hidden_input, self.W1[cal_items]).reshape(masked_input.shape[0],self.cat_num) + self.B1[cal_items]
        prediction = pred.sigmoid().log().reshape(samples_include_neg.shape[0], samples_include_neg.shape[1], self.cat_num)
#         pdb.set_trace()
        prediction = torch.cat([prediction[:,:(hist.shape[1]+1),1], prediction[:,(hist.shape[1]+1):,0]],dim=1)
        format_pred = torch.zeros_like(one_hot_1).type(torch.float).scatter_(1,samples_include_neg, prediction)
        
#         pdb.set_trace()
        
        
        out_dict = {'prediction': pred, 'format_pred': format_pred}
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
            out_dict['format_pred'][node]=0
            
        out_dict['loss'] = -out_dict['format_pred'].mean()
        out_dict['config'] = config
        return out_dict
    
        
        