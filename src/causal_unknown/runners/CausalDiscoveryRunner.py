import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as opt
from time import time
from runners.BaseRunner import BaseRunner
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from collections import defaultdict
from utils import evaluator
from utils import utils
import pdb

class CausalDiscoveryRunner(BaseRunner):
    def parse_runner_args(parser):
        parser.add_argument('--gopt', default='Adam', type=str,
                           help='Select gamma optimizer')
        parser.add_argument('--glr', default=0.001, type=float,
                           help='learning rate of gamma optimizer')
        parser.add_argument('--gl2', default=1e-5, type=float,
                           help='weight of l2-regularizer for gamma optimizer')
        parser.add_argument("--lsparse", default=0.0 ,type=float,
                           help="Regularizer for sparsity.")
        parser.add_argument("--lmaxent", default=0.0 ,type=float,
                           help="Regularizer for maximum entropy.")
        parser.add_argument("--ldag", default=0.1 ,type=float,
                           help="Regularizer for DAGness.")
        
        return BaseRunner.parse_runner_args(parser)
        
        
    def __init__(self, gopt, glr, gl2, lsparse, lmaxent, ldag, *args, **kwargs):
        BaseRunner.__init__(self, *args, **kwargs)
        self.gopt_name = gopt.lower()
        self.glr = glr
        self.gl2 = gl2
        self.goptimizer = None
        self.lsparse = lsparse
        self.lmaxent = lmaxent
        self.ldag = ldag
        
    def _build_optimizer(self, model):
        BaseRunner._build_optimizer(self, model)
            
        if self.gopt_name == 'sgd':
            self.goptimizer = opt.SGD(model.structural_parameters(), lr=self.glr)
        elif self.gopt_name == 'adagrad':
            self.goptimizer = opt.Adagrad(model.structural_parameters(), lr=self.glr)
        elif self.gopt_name == 'adam':
            self.goptimizer = opt.Adam(model.structural_parameters(), lr=self.glr)
        else:
            print("Unknown Gamma Optimizer: " + gopt)
            self.goptimizer = opt.SGD(model.structural_parameters(), lr=self.glr)
            
    def fit(self, model, data):
        """
        phrase 1
        """
        losses = []
        data.get_data(intervention=False)
        dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        model.train()
        pbar = tqdm(total=len(data), leave=False, ncols=100, mininterval=1, desc='Training observation')
        for i, batchData in enumerate(dataLoader):
            pbar.update(batchData['uid'].shape[0])
            self.optimizer.zero_grad()
            out_dict = model(batchData, True)
            loss = out_dict['loss'] + model.l2() * self.l2
            losses.append(loss.detach().cpu())
            loss.backward()
#             torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            self.optimizer.step()
        pbar.close()
        model.eval()
        return np.mean(losses)
    
    def fit_intervention(self, model, data):
        """
        phrase 2 and 3
        """
        
        losses = []
        data.get_data(intervention=True)
        dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=16)
        model.train()
        
        self.goptimizer.zero_grad()
        model.gamma.grad = torch.zeros_like(model.gamma)
        gammagrads = [] # List of T tensors of shape (M,M,) indexed by (i,j)
        logregrets = []
        gammasigmoid = model.gamma.sigmoid()
        
        pbar = tqdm(total=len(data), leave=False, ncols=100, mininterval=1, desc='Training intervention')
        for i, batchData in enumerate(dataLoader):
            pbar.update(batchData['uid'].shape[0])
            self.optimizer.zero_grad()
            out_dict = model(batchData, intervention=True)
            
            logpn = out_dict['format_pred']
            gammagrad = gammasigmoid - out_dict['config']
            logregret = logpn
            
            gammagrads.append(gammagrad)
            logregrets.append(logregret)
            
            loss = out_dict['loss'] + model.l2() * self.l2
            losses.append(loss.detach().cpu())
            loss.backward()
            self.optimizer.step()
        pbar.close()
        model.eval()
        
        with torch.no_grad():
            gammagrads = torch.stack(gammagrads)
            logregrets = torch.stack(logregrets)
            normregret = logregrets.softmax(0)
            dRdgamma   = torch.einsum("kij,ki->ij", gammagrads, normregret)
            model.gamma.grad.copy_(dRdgamma)
            
        siggamma = model.gamma.sigmoid()
        Lmaxent  = ((siggamma)*(1-siggamma)).sum().mul(-self.lmaxent)
        Lsparse  = siggamma.sum().mul(float(self.lsparse))
        Ldag     = siggamma.mul(siggamma.t()).cosh().tril(-1).sum() \
                           .sub(model.item_num**2 - model.item_num) \
                           .mul(self.ldag)
        (Lmaxent + Lsparse + Ldag).backward()

        """Perform Gamma Update with constraints"""
        self.goptimizer.step()
        model.reconstrain_gamma()
        
        return np.mean(losses)
            
    def train(self, model, trainset, validationset, testset):
        if self.optimizer is None:
            self._build_optimizer(model)
        self._check_time(start=True)
        
        
#         init_valid = self.evaluate(model, validationset)
#         init_test = self.evaluate(model, testset)
        
#         logging.info('Init: \t validation= %s test= %s [%.1f s]' % (init_valid[1], init_test[1], self._check_time()) + ','.join(self.metrics))
        
        
        for epoch in range(self.epoch):
            
            self._check_time()
            trainset.get_data(intervention=False)
            trainset.get_neg(testset.hist_dict)
            
            
            train_loss = self.fit(model, trainset)
#             valid = self.evaluate(model, validationset)
            
            trainset.get_data(intervention=True)
            trainset.get_neg(testset.hist_dict)
            train_loss += self.fit_intervention(model, trainset)
            
            train_time = self._check_time()
            
            valid = self.evaluate(model, validationset)
            test = self.evaluate(model, testset)
            test_time = self._check_time()
            
            self.train_results.append(train_loss)
            self.valid_results.append(valid[0][0])
            self.test_results.append(test[0][0])
            self.valid_msg.append(valid[1])
            self.test_msg.append(test[1])
            
            logging.info("Epoch %5d [%.1f s] \t train= %s validation= %s test= %s [%.1f s]" 
                         % (epoch + 1, train_time, str(train_loss), valid[1], test[1], test_time) + ','.join(self.metrics))
            if self.valid_results[-1] == max(self.valid_results):
                model.save_model()
            if self.eva_terminaion() and self.early_stop == 1:
                logging.info("Early stop at %d based on validation result" % (epoch + 1))
                break
            logging.info("")
                
        best_valid_eval = max(self.valid_results)
        best_valid_epoch = self.valid_results.index(best_valid_eval)
        logging.info("Best Iteration (validation)= %5d \t validation= %s test= %s" 
                     % (best_valid_epoch + 1, self.valid_msg[best_valid_epoch], self.test_msg[best_valid_epoch]))
        
        best_test_eval = max(self.test_results)
        best_test_epoch = self.test_results.index(best_test_eval)
        logging.info("Best Iteration (test)= %5d \t validation= %s test= %s" 
                     % (best_test_epoch + 1, self.valid_msg[best_test_epoch], self.test_msg[best_test_epoch]))
        model.load_model()
        
        
        