from models.BaseModel import BaseModel
import torch
import torch.nn.functional as F
import pdb

class MF(BaseModel):
    loader = 'MatchingLoader'
    runner = 'PointwiseRunner'
    
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
        return self.predict(batch)
    
    def forward(self, batch):
        """
        calculate the loss
        """
        out_dict = self.estimate(batch)
        loss = self.loss_func(out_dict['label'].to(torch.float), out_dict['prediction'])
        out_dict['loss'] = loss
        return out_dict