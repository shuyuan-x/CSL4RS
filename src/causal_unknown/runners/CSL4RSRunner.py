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
from utils.dag import compute_dag_constraint, compute_penalty, TrExpScipy, is_acyclic
import pdb

class CSL4RSRunner(BaseRunner):
    def parse_runner_args(parser):
        parser.add_argument('--mu0', default=1e-8, type=float,
                           help='coefficient of augmented langrangian.')
        parser.add_argument("--lambda0", default=0.1 ,type=float,
                           help="Regularizer for DAGness.")
        parser.add_argument('--delta', default=0.9, type=float,
                           help='coefficient to determine whether update mu')
        parser.add_argument('--gamma', default=2.0, type=float,
                           help='coefficient to update mu')
        
        return BaseRunner.parse_runner_args(parser)
    
    def __init__(self, mu0, lambda0, delta, gamma, *args, **kwargs):
        BaseRunner.__init__(self, *args, **kwargs)
        self.mu_t = mu0
        self.lambda_t = lambda0
        self.delta = delta
        self.gamma = gamma
        self.acyclic_results = []
        
        
    def fit(self, model, data):
        losses = []
        
        
        dataLoader = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=8)
        model.train()
        pbar = tqdm(total=len(data), leave=False, ncols=100, mininterval=1, desc='Predict')
        for i, batchData in enumerate(dataLoader):
            pbar.update(batchData['uid'].shape[0])
            self.optimizer.zero_grad()
            out_dict = model(batchData)
            
            # constraint related
            w_adj = model.get_w_adj()
            h = compute_dag_constraint(w_adj)
            constraint_violation = h.item()
            
            # compute regularizer
            reg = self.l2 * compute_penalty([w_adj], p=1)
            reg = reg / (w_adj.shape[0] ** 2)
            
#             pdb.set_trace()
            # compute augmented langrangian
            lagrangian = out_dict['loss'] + reg + self.lambda_t * h
            augmentation = h ** 2
#             pdb.set_trace()
            
            aug_lagrangian = lagrangian + 0.5 * self.mu_t * augmentation
            
            losses.append(aug_lagrangian.detach().cpu())
            aug_lagrangian.backward()
            self.optimizer.step()
        pbar.close()
        model.eval()
        return np.mean(losses)
    
    def train(self, model, trainset, validationset, testset):
        if self.optimizer is None:
            self._build_optimizer(model)
        self._check_time(start=True)
        
        h_values = []
        w_adj = model.get_w_adj()
        h = compute_dag_constraint(w_adj)
        constraint_violation = h.item()
        h_values.append(constraint_violation)
        
        for epoch in range(self.epoch):
            self._check_time()
            trainset.get_neg(testset.hist_dict)
            
            train_loss = self.fit(model, trainset)
            train_time = self._check_time()
            
            w_adj = model.get_w_adj()
            h = compute_dag_constraint(w_adj)
            constraint_violation = h.item()
            
            self.lambda_t += self.mu_t * constraint_violation
            if constraint_violation > self.delta * h_values[-1]:
                self.mu_t = self.mu_t * self.gamma
            
            valid = self.evaluate(model, validationset)
#             test = self.evaluate(model, testset)
            test_time = self._check_time()
            
            self.train_results.append(train_loss)
            self.valid_results.append(valid[0][0])
#             self.test_results.append(test[0][0])
            self.valid_msg.append(valid[1])
#             self.test_msg.append(test[1])
            
            with torch.no_grad():
                to_keep = (model.get_w_adj() > 0.5).type(torch.Tensor).to(model.adj.device)
                current_adj = model.adj * to_keep
                current_adj = current_adj.cpu().numpy()
                acyclic = is_acyclic(current_adj)
            self.acyclic_results.append(acyclic)
            
            acyclic_valid = [a * b for a,b in zip(self.valid_results, self.acyclic_results)]
            
#             logging.info("Epoch %5d [%.1f s] \t train= %s validation= %s test= %s acyclic=%s [%.1f s]" 
#                          % (epoch + 1, train_time, str(train_loss), valid[1], test[1], str(acyclic), test_time) + ','.join(self.metrics))
            logging.info("Epoch %5d [%.1f s] \t train= %s validation= %s acyclic=%s [%.1f s]" 
                         % (epoch + 1, train_time, str(train_loss), valid[1], str(acyclic), test_time) + ','.join(self.metrics))
            if self.valid_results[-1] == max(acyclic_valid) and acyclic:
                model.save_model()
            if self.eva_terminaion() and self.early_stop == 1:
                logging.info("Early stop at %d based on validation result" % (epoch + 1))
                break
            logging.info("")
                
#         acyclic_valid = [a * b for a,b in zip(self.valid_results, self.acyclic_results)]
#         best_valid_eval = max(acyclic_valid)
#         best_valid_epoch = acyclic_valid.index(best_valid_eval)
#         logging.info("Best Iteration (validation)= %5d \t validation= %s test= %s" 
#                      % (best_valid_epoch + 1, self.valid_msg[best_valid_epoch], self.test_msg[best_valid_epoch]))
        
        model.load_model()
        test = self.evaluate(model, testset)
        acyclic_valid = [a * b for a,b in zip(self.valid_results, self.acyclic_results)]
        best_valid_eval = max(acyclic_valid)
        best_valid_epoch = acyclic_valid.index(best_valid_eval)
        logging.info("Best Iteration (validation)= %5d \t validation= %s test= %s" 
                     % (best_valid_epoch + 1, self.valid_msg[best_valid_epoch], test[1]))
        
#         acyclic_test = [a * b for a,b in zip(self.test_results, self.acyclic_results)]
#         best_test_eval = max(acyclic_test)
#         best_test_epoch = acyclic_test.index(best_test_eval)
#         logging.info("Best Iteration (test)= %5d \t validation= %s test= %s" 
#                      % (best_test_epoch + 1, self.valid_msg[best_test_epoch], self.test_msg[best_test_epoch]))
        model.load_model()
        
#     def eva_terminaion(self):
#         """
#         whether stop the training, based on the validation evaluation
#         @output:
#         - return: stop the training or not, True or False
#         """
#         valid = self.valid_results
        
#         if len(valid) > 20 and utils.strictly_decreasing(valid[-5:]):
#             return True
#         elif len(valid) - valid.index(max(valid)) > 20:
#             return True
#         return False