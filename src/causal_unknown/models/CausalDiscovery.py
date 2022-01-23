from models.BaseModel import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
import math
import pdb


class CausalDiscovery(BaseModel):
    loader = 'CausalDiscoveryLoader'
    runner = 'CausalDiscoveryRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--hidden_size', default=4, type=int,
                           help='Size of hidden vectors')
        return BaseModel.parse_model_args(parser)
    
    def __init__(self, hidden_size, *args, **kwargs):
        self.hidden_size = hidden_size
        BaseModel.__init__(self, *args, **kwargs)
        
    def _init_weight(self):
        self.cat_num = 2
        Ns = self.item_num * self.cat_num
        M = self.item_num
        H = self.hidden_size
        self.W0 = torch.nn.Parameter(torch.empty(M, Ns, H))
        self.B0 = torch.nn.Parameter(torch.empty(M, H))
        self.W1 = torch.nn.Parameter(torch.empty(M, H, self.cat_num))
        self.B1 = torch.nn.Parameter(torch.empty(M, self.cat_num))
#         with torch.no_grad():
#             a = 1.0 / math.sqrt(M)
#             self.W0.normal_(mean=0.0, std=a)
#             self.W1.normal_(mean=0.0, std=a)
#             self.B0.normal_(mean=0.0, std=a)
#             self.B1.normal_(mean=0.0, std=a)
        torch.nn.init.uniform_(self.W0, b=0.1)
        torch.nn.init.uniform_(self.W1, b=0.1)
        torch.nn.init.uniform_(self.B0, b=0.1)
        torch.nn.init.uniform_(self.B1, b=0.1)
        self.gamma = torch.nn.Parameter(torch.empty(M, M))
        with torch.no_grad(): self.gamma.diagonal().fill_(float("-inf"))
        self.LeakyRelu = torch.nn.LeakyReLU(0.1)
#         pdb.set_trace()
        
    def parameters(self):
        s = set(self.structural_parameters())
        l = [p for p in super().parameters() if p not in s]
        return iter(l)
    
    def structural_parameters(self):
        return iter([self.gamma])
        
#     def _init_gamma(self):
#         self.gamma = torch.empty((self.item_num, self.item_num),dtype=torch.float32)
#         expParents = self.p
#         idx        = np.arange(self.item_num).astype(np.float32)[:,np.newaxis]
#         idx_maxed  = np.minimum(idx*0.5, expParents)
#         p          = np.broadcast_to(idx_maxed/(idx+1), (self.item_num, self.item_num))
#         B          = np.random.binomial(1, p)
#         B          = np.tril(B, -1)
#         self.gamma.copy_(torch.as_tensor(B))
#         return
    
    def predict(self, batch):
        uids = batch['uid'].to(torch.long).to(self.W0.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.W0.device)
        hist = batch['history'].to(torch.long).to(self.W0.device)
        label = batch['rating'].to(torch.long).to(self.W0.device).view([-1])
#         pdb.set_trace()
        with torch.no_grad():
            gammaexp = self.gamma.sigmoid().gt_(torch.tensor(0.5).to(self.W0.device))
#             self.gamma.sigmoid()
            gammaexp.diagonal().zero_()
            
        mask = torch.repeat_interleave(gammaexp[iids], self.cat_num, dim=1)
        one_hot_1 = F.one_hot(hist, num_classes=self.item_num).sum(dim=1).gt_(torch.tensor(0).to(self.W0.device))
        one_hot_0 = torch.ones_like(one_hot_1) - one_hot_1
        one_hot = torch.cat([one_hot_0.t(), one_hot_1.t()], dim=-1).view(-1, one_hot_0.shape[0]).t()
        
        masked_input = mask.to(torch.float) * one_hot.to(torch.float)
        format_input = masked_input.view(masked_input.shape[0], 1, masked_input.shape[1])
        hidden_pred = torch.matmul(format_input, self.W0[iids]).reshape(masked_input.shape[0],self.hidden_size) + self.B0[iids]
        hidden_input = self.LeakyRelu(hidden_pred).view(hidden_pred.shape[0], 1, hidden_pred.shape[1])
#         pdb.set_trace()
        pred = torch.matmul(hidden_input, self.W1[iids]).reshape(masked_input.shape[0],self.cat_num) + self.B1[iids]
        prediction = pred.sigmoid()[:,1].view([-1])
        
        out_dict = {'prediction': prediction, 'uid': uids, 'label': label}
        return out_dict
        
        
    def gl2(self):
        l2 = 0
        for p in self.structural_parameters():
            l2 += (p ** 2).sum()
            
        return l2
            
    def estimate(self, batch, config):
        uids = batch['uid'].to(torch.long).to(self.W0.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.W0.device)
        hist = batch['history'].to(torch.long).to(self.W0.device)
#         negs = batch['negative'].to(torch.long).to(self.W0.device).view([-1])
        
        mask = torch.repeat_interleave(config, self.cat_num, dim=1)
        samples = torch.cat([hist, iids.reshape(iids.shape[0], 1)], dim=1)
        one_hot_1 = F.one_hot(samples, num_classes=self.item_num).sum(dim=1).gt_(torch.tensor(0).to(self.W0.device))
        one_hot_0 = torch.ones_like(one_hot_1) - one_hot_1
        one_hot = torch.cat([one_hot_0.t(), one_hot_1.t()], dim=-1).view(-1, one_hot_0.shape[0]).t()
        
        
        one_hot = one_hot.view(one_hot.shape[0],1,one_hot.shape[1]).expand(one_hot.shape[0],self.item_num,one_hot.shape[1])
        mask = mask.view(1, mask.shape[0], mask.shape[1]).expand(iids.shape[0], mask.shape[0], mask.shape[1])
        masked_input = mask.to(torch.float) * one_hot.to(torch.float)
        format_input = masked_input.view(masked_input.shape[0], masked_input.shape[1], 1, masked_input.shape[2])
        
        
        hidden_pred = torch.matmul(format_input, self.W0).reshape(masked_input.shape[0],masked_input.shape[1],self.hidden_size)
        hidden_pred += self.B0.view(1, self.B0.shape[0], self.B0.shape[1]).expand(hidden_pred.shape)
        hidden_input = self.LeakyRelu(hidden_pred).view(hidden_pred.shape[0], hidden_pred.shape[1], 1, hidden_pred.shape[2])
#         pdb.set_trace()
        pred = torch.matmul(hidden_input, self.W1).reshape(hidden_input.shape[0], hidden_input.shape[1], self.cat_num)
        pred += self.B1.view(1, self.B1.shape[0], self.B1.shape[1]).expand(pred.shape)
        prediction = pred.sigmoid()
        
        indicator = torch.cat([one_hot_0.reshape(iids.shape[0],self.item_num,1),
                               one_hot_1.reshape(iids.shape[0],self.item_num,1)],dim=2)
        
        prob = (indicator.to(torch.float) * prediction).sum(dim=-1).log()
        
        out_dict = {'prediction': prob}
        return out_dict
    
    def forward(self, batch, intervention=False):
        with torch.no_grad():
            config = self.gamma.sigmoid()
            config = torch.empty_like(config).uniform_().lt_(config)
            config.diagonal().zero_()
        
        out_dict = self.estimate(batch, config)
        out_dict['prediction'][torch.isnan(out_dict['prediction'])] = 0
        out_dict['prediction'][out_dict['prediction'] == float("inf")] = 20
        out_dict['prediction'][out_dict['prediction'] == float("-inf")] = -20
        if intervention:
            node = torch.argmax(-out_dict['prediction'].mean(0)).item()
#             pdb.set_trace()
            out_dict['prediction'][:,node]=0
            out_dict['format_pred'] = out_dict['prediction'].mean(0)
        out_dict['loss'] = -out_dict['prediction'].mean()
#         pdb.set_trace()
        out_dict['config'] = config
        return out_dict
    
    def reconstrain_gamma(self):
        with torch.no_grad():
            self.gamma.clamp_(-5,+5)
            self.gamma.diagonal().fill_(float("-inf"))
        
        