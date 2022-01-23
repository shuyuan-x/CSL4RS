from models.BaseModel import BaseModel
import torch
import torch.nn.functional as F
import pdb

class GRU4Rec(BaseModel):
    loader = 'HistLoader'
    runner = 'BaseRunner'
    
    def parse_model_args(parser):
        parser.add_argument('--hidden_size', default=64, type=int,
                           help='Size of hidden vectors in GRU')
        parser.add_argument('--num_layers', default=1, type=int,
                           help='Number of GRU layers')
        parser.add_argument('--emb_size', default=64, type=int,
                           help='Size of embedding')
        return BaseModel.parse_model_args(parser)
    
    def __init__(self, hidden_size, num_layers, emb_size, *args, **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_size = emb_size
        BaseModel.__init__(self, *args, **kwargs)
        
    def _init_weight(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.emb_size)
        self.rnn = torch.nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.out = torch.nn.Linear(self.hidden_size, self.emb_size, bias=False)
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        
        return
    
    def predict(self, batch):
        """
        prediction for evaluation
        """
        
        uids = batch['uid'].to(torch.long).to(self.dummy_param.device).view([-1])
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        label = batch['rating'].to(torch.long).to(self.dummy_param.device).view([-1])
        pad_his = batch['history'].to(torch.long).to(self.dummy_param.device)

#         hist = [torch.tensor([int(i) for i in row.split(',')]) for row in batch['history']]
#         pad_his = torch.nn.utils.rnn.pad_sequence(hist, batch_first=True).to(self.dummy_param.device)
        
        valid_his = pad_his.gt(0).long()
        his_length = valid_his.sum(dim=-1)
        
        his_vectors = self.iid_embeddings(pad_his) * valid_his.unsqueeze(dim=-1).float()
        
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=pad_his.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        
        # pack
        packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        
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
        
    def estimate(self, batch):
        """
        estimation for training
        """
        iids = batch['iid'].to(torch.long).to(self.dummy_param.device)
        negs = batch['negative'].to(torch.long).to(self.dummy_param.device).view([-1])
        pad_his = batch['history'].to(torch.long).to(self.dummy_param.device)
        
#         hist = [torch.tensor([int(i) for i in row.split(',')]) for row in batch['history']]
#         pad_his = torch.nn.utils.rnn.pad_sequence(hist, batch_first=True).to(self.dummy_param.device)
        
        valid_his = pad_his.gt(0).long()
        his_length = valid_his.sum(dim=-1)
        
        his_vectors = self.iid_embeddings(pad_his) * valid_his.unsqueeze(dim=-1).float()
        
        
        # sort
        sorted_his_length, sorted_idx = torch.topk(his_length, k=pad_his.shape[0])
        sorted_his_vectors = his_vectors.index_select(dim=0, index=sorted_idx)
        
        # pack
        packed_his_vectors = torch.nn.utils.rnn.pack_padded_sequence(sorted_his_vectors, sorted_his_length.cpu(), batch_first=True)
        
        # rnn
        out_put, hidden = self.rnn(packed_his_vectors, None)
        
        sorted_rnn_vector = self.out(hidden[-1])
        
        # unsort
        unsort_idx = torch.topk(sorted_idx, k=pad_his.shape[0], largest=False)[1]
        rnn_vec = sorted_rnn_vector.index_select(dim=0, index=unsort_idx)
        
        # predict
        pos_item_vec = self.iid_embeddings(iids)
        pos_prediction = (rnn_vec * pos_item_vec).sum(dim=1).view([-1])
        neg_item_vec = self.iid_embeddings(negs)
        neg_prediction = (rnn_vec * neg_item_vec).sum(dim=1).view([-1])
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
        