# GRU4Rec
python main.py --model GRU4Rec --val_neg 100 --test_neg 100 --epoch 100 --batch_size 1024 --gpu 0 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --dataset Electronics_15_15

# item2vec
python main.py --model item2vec --opt adam --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --dataset Electronics_15_15

# NOTEARS
python main.py --model notears --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --dataset Electronics_15_15 --mu0 1e-15

# SDI
python main.py --model SDI --epoch 10 --val_neg 100 --test_neg 100 --batch_size 32 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --intervention_ratio 0.5 --dataset Electronics_15_15

# CSL4RS-L
python main.py --model CSL4RSLinear --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15

# CSL4RS Abalation (no GRU)
python main.py --model CSL4RSMLP --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15

# CSL4RS
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15

# Sensitivity Analysis
# a) MLP hidden dimension (default is 4)
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_MLP 2
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_MLP 8
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_MLP 16
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_MLP 32

# b) GRU hidden dimension (default is 64)
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_GRU 4
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_GRU 8
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_GRU 16
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --hidden_size_GRU 32

# c) initial penalty parameter lambda (default is 0.1)
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --lambda0 1e-5
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --lambda0 1e-3
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --lambda0 0.5
python main.py --model CSL4RS --opt rmsprop --epoch 100 --val_neg 100 --test_neg 100 --batch_size 1024 --gpu 1 --max_hist 5 --metrics hit@1,hit@5,ndcg@5,mrr --eval_batch_size 30000 --mu0 1e-15 --dataset Electronics_15_15 --lambda0 1.0