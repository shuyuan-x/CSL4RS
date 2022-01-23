import torch
import pdb

def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)

def gumbel_sigmoid(log_alpha, uniform, batch_target, tau=1):
    shape = tuple([batch_target.shape[0],log_alpha.shape[1]])
    logistic_noise = sample_logistic(shape, uniform)
    
    y_soft = torch.sigmoid((log_alpha[batch_target] + logistic_noise.to(log_alpha.device)) / tau)
    
    y_hard = y_soft.clone().gt_(torch.tensor(0.5))
    
#     pdb.set_trace()
    y = y_hard.detach().to(y_soft.device) - y_soft.detach() + y_soft
    
    
    return y