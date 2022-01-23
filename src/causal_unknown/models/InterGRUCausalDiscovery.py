from models.GRUCausalDiscovery import GRUCausalDiscovery
import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb

class InterGRUCausalDiscovery(GRUCausalDiscovery):
    loader = 'HistLoader'
    runner = 'InterCausalDiscoveryRunner'
    
    
    def forward(self, batch):
        with torch.no_grad():
            config = self.gamma.sigmoid()
            config = torch.empty_like(config).uniform_().lt_(config)
            config.diagonal().zero_()
        
        out_dict = self.estimate(batch, config)
#         pdb.set_trace()
        
        out_dict['format_pred'] = out_dict['format_pred'].sum(0) / (out_dict['format_pred']!=0).sum(0)
        out_dict['format_pred'][torch.isnan(out_dict['format_pred'])] = 0
        node = torch.argmax(-out_dict['format_pred']).item()
        out_dict['format_pred'][node]=0#out_dict['format_pred'][node]/2
            
        out_dict['loss'] = -out_dict['format_pred'].mean()
        out_dict['config'] = config
        return out_dict