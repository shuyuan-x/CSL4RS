import torch
from sklearn.metrics import *
import numpy as np
import math
import pdb

def evaluate(pred, gt, metrics):
#     pdb.set_trace()
    evaluation = []
    sorted_p, sorted_gt = sort(pred, gt)
#     pdb.set_trace()
    for metric in metrics:
        if metric == 'mrr':
#             pdb.set_trace()
            evaluation.append(mean_reciprocal_rank(sorted_gt))
        else:
            k = int(metric.split('@')[-1])
            if metric.startswith('hit@'):
                evaluation.append(hit_at_k(sorted_gt, k))
            elif metric.startswith('precision@'):
                evaluation.append(precision_at_k(sorted_gt, k))
            elif metric.startswith('recall@'):
                evaluation.append(recall_at_k(sorted_gt, k))
            elif metric.startswith('ndcg@'):
                evaluation.append(ndcg_at_k(sorted_gt, k))
            
    format_str = []
    for m in evaluation:
        format_str.append('%.4f' % m)
#     pdb.set_trace()
    return evaluation, ','.join(format_str)
    
def sort(pred, gt):
    sorted_p, sorted_gt = {}, {}
    for i in pred:
#         pdb.set_trace()
        index = np.argsort(-np.array(pred[i]))
        sorted_p[i] = np.array(pred[i])[index]
        sorted_gt[i] = np.array(gt[i])[index]
    return sorted_p, sorted_gt

def hit_at_k(sorted_gt, k):
    hit = 0.0
    for user in sorted_gt:
        if np.sum(sorted_gt[user][:k]) > 0:
            hit += 1
    return hit/len(sorted_gt)

def precision_at_k(sorted_gt, k):
    pre = 0.0
    for user in sorted_gt:
        pre += np.sum(sorted_gt[user][:k])/k
    return pre/len(sorted_gt)

def recall_at_k(sorted_gt, k):
    recall = 0.0
    for user in sorted_gt:
        recall += np.sum(sorted_gt[user][:k]) / np.sum(sorted_gt[user])
    return recall / len(sorted_gt)

def ndcg_at_k(sorted_gt, k):
    ndcg = 0.0
    for user in sorted_gt:
        dcg = 0.0
        idcg = 0.0
        for i in range(k):
            if sorted_gt[user][i] > 0:
                dcg += 1. / np.log2(i + 2)
        for i in range(min(k, np.sum(sorted_gt[user]))):
            idcg += 1. / np.log2(i + 2)
        ndcg += dcg / idcg
    return ndcg / len(sorted_gt)

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rank = [gt.nonzero()[0] for gt in rs.values()]
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rank])
