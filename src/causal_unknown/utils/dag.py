import torch
import numpy as np
from scipy.linalg import expm
from utils.gumbel import gumbel_sigmoid
import pdb

class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix
    """
    
    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                expm_input = expm_input.to(input.device)
                assert expm_input.is_cuda
                
            ctx.save_for_backward(expm_input)
            
        return torch.trace(expm_input)
    
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            expm_input, = ctx.saved_tensors
            return expm_input.t() * grad_output
        
def compute_dag_constraint(w_adj):
    """
    compute the DAG constraint of w_adj
    """
    try:
        assert (w_adj >= 0).detach().cpu().numpy().all()
    except:
        pdb.set_trace()
    h = TrExpScipy.apply(w_adj) - w_adj.shape[0]
    return h

def is_acyclic(adj):
    """
    return True if adj is a acyclic
    """
    prod = np.eye(adj.shape[0])
    for _ in range(1, adj.shape[0]+1):
        prod = np.matmul(adj, prod)
        if np.trace(prod) != 0: return False
    return True
    
def compute_penalty(list_, p=2, target=0.):
    penalty = 0
    for m in list_:
        penalty += torch.norm(m-target, p=p) ** p
    return penalty
    
class GumbelAdjacency(torch.nn.Module):
    def __init__(self, num_vars):
        super(GumbelAdjacency, self).__init__()
        self.num_vars = num_vars
        self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))
        self.uniform = torch.distributions.uniform.Uniform(0,1)
        self.reset_parameters()
        
    def forward(self, batch_target, tau=1):
        adj = gumbel_sigmoid(self.log_alpha, self.uniform, batch_target, tau=tau)
        return adj
    
    def get_prob(self):
        return torch.sigmoid(self.log_alpha)
    
    def reset_parameters(self):
        torch.nn.init.uniform(self.log_alpha, -5.0, -2.0)
#         torch.nn.init.constant_(self.log_alpha, -5)
        