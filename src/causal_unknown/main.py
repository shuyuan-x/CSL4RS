from utils import utils
from loaders.HistLoader import HistLoader
from loaders.BaseLoader import BaseLoader
from loaders.SDILoader import SDILoader
from models.GRU4Rec import GRU4Rec
from models.item2vec import item2vec
from models.notears import notears
from models.CSL4RS import CSL4RS
from models.CSL4RSLinear import CSL4RSLinear
from models.CSL4RSMLP import CSL4RSMLP
from models.SDI import SDI
from runners.BaseRunner import BaseRunner
from runners.CSL4RSRunner import CSL4RSRunner
from runners.SDIRunner import SDIRunner
import argparse
import logging
import sys
import torch
import numpy as np
import os
import pdb

def main():
    init_parser = argparse.ArgumentParser(description='Indicate Model')
    init_parser.add_argument('--model', default='CategoricalWorld', type=str,
                             help='the name of model')
    init_parser.add_argument('--runner', default='BaseRunner', type=str,
                             help='the name of runner')
    init_parser.add_argument('--dataloader', default='BaseLoader', type=str,
                             help='the name of dataloader')
    init_args, init_extra_args = init_parser.parse_known_args()
    
    
    model_name = eval(init_args.model)
    init_args.runner = model_name.runner
    runner_name = eval(init_args.runner)
    init_args.dataloader = model_name.loader
    loader_name = eval(init_args.dataloader)
    
    parser = utils.parse_global_args()
    parser = model_name.parse_model_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = loader_name.parse_loader_args(parser)
    
    
    args, extra_args = parser.parse_known_args()
    
    paras = sorted(vars(args).items(), key=lambda kv: kv[0])
    
    log_name_exclude = ['verbose', 'gpu', 'seed', 'dataset', 'path', 'pt', 
                        'model_path', 'log_file', 'metrics', 'load', 'train', 'eval_batch_size', 
                        'early_stop']
    
    log_file_name = [str(args.pt), str(init_args.model), str(args.dataset), str(args.seed)] + \
                    [p[0].replace('_','')[:3] + str(p[1]) for p in paras if p[0] not in log_name_exclude]
    log_file_name = [l.replace(' ','-').replace('_', '-') for l in log_file_name]
    log_file_name = '_'.join(log_file_name)
    
    
    args.log_file = os.path.join('../../log/', '%s/%s/%s.txt' % (init_args.model, args.dataset, log_file_name))
    utils.check_dir_and_mkdir(args.log_file)
    args.model_path = os.path.join('../../model/', '%s/%s/%s.pt' % (init_args.model, args.dataset, log_file_name))
    utils.check_dir_and_mkdir(args.model_path)
    
    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info('=======================================')
    logging.info(vars(init_args))
    logging.info(vars(args))
    
    logging.info('DataLoader: ' + init_args.dataloader)
    logging.info('Model: ' + init_args.model)
    logging.info('Runner: ' + init_args.runner)

    
    
    # random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.info('# cuda devices: %d' % torch.cuda.device_count())
    
    
    # create dataset
    dl_para_dict = utils.get_init_paras_dict(loader_name, vars(args))
    logging.info(init_args.dataloader + ': ' + str(dl_para_dict))
    trainset = loader_name(**dl_para_dict)
    trainset.set_task('train')
    validationset = loader_name(**dl_para_dict)
    validationset.set_task('validation')
    testset = loader_name(**dl_para_dict)
    testset.set_task('test')
    
    if init_args.dataloader in ['HistLoader', 'SDILoader', 'GRUCausalDiscoveryLoader']:
        trainset.get_hist()
        validationset.get_hist(trainset.hist_dict)
        testset.get_hist(validationset.hist_dict)
        trainset.get_neg(testset.hist_dict)
        validationset.get_neg(testset.hist_dict)
        testset.get_neg(testset.hist_dict)
    elif init_args.dataloader in ['MatchingLoader']:
        trainset.get_hist()
        validationset.get_hist(trainset.hist_dict)
        testset.get_hist(validationset.hist_dict)
        validationset.get_neg_eval(testset.hist_dict)
        testset.get_neg_eval(testset.hist_dict)
    logging.info('# users: ' + str(trainset.user_num))
    logging.info('# items: ' + str(trainset.item_num))
        
    # create model    
    loader_vars = vars(trainset)
    for key in loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = loader_vars[key]
    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(init_args.model + ': ' + str(model_para_dict))
    model = model_name(**model_para_dict)
    
    # create runner
    runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
    logging.info(init_args.runner + ': ' + str(runner_para_dict))
    runner = runner_name(**runner_para_dict)
    
    if torch.cuda.device_count() > 0:
        model = model.cuda()
        
    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(model, trainset, validationset, testset)
    
    
if __name__ == '__main__':
    main()
    
    
