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


class PointwiseRunner(BaseRunner):
    def train(self, model, trainset, validationset, testset):
        if self.optimizer is None:
            self._build_optimizer(model)
        self._check_time(start=True)
        
        init_valid = self.evaluate(model, validationset)
        init_test = self.evaluate(model, testset)
        
        logging.info('Init: \t validation= %s test= %s [%.1f s]' % (init_valid[1], init_test[1], self._check_time()) + ','.join(self.metrics))
        
        for epoch in range(self.epoch):
            self._check_time()
            
            train_loss = self.fit(model, trainset)
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