import torch
import numpy as np
import random
import time
import argparse
from data_kit import extractor_feature, extractor_feature_test
from extractor import FeatureExtractor
from framework import test


torch.cuda.set_device(3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/data/train_wiki.json',
                        help='training set')    
    parser.add_argument('--test_path', default='/data/test_wiki_input-5-1.json', 
                        help='test set')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Q')
    parser.add_argument('--num_task', default=1000, type=int,
                        help='number of task')
    parser.add_argument('--sampled', default=700, type=int,
                        help='sampled number per instance')
    parser.add_argument('--k', default=5, type=int,
                        help='k nearest classes')
    parser.add_argument('--p', default=0.1, type=float,
                        help='calibrated mean rate p')
    parser.add_argument('--alpha', default=0.25, type=float,
                        help='calibrated cov extra alpha')                        
    parser.add_argument('--classifier', default='LR', 
                        help='classifier type LR or SVM')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed')
    parser.add_argument('--ckpt', default='/ckpt/pretrained_model', 
                        help='checkpoint path')

    opt = parser.parse_args()
    N = opt.N
    K = opt.K
    Q = opt.Q
    num_task = opt.num_task
    sampled_num = opt.sampled
    k = opt.k
    p = opt.p
    alpha = opt.alpha
    seed = opt.seed
    classifier = opt.classifier
    base_data_path = opt.train_path
    test_data_path = opt.val_path
    ckpt_path = opt.ckpt

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    base_feature_path = 'feature/base_faeture.json'
    test_feature_path = 'feature/test_feature.json'
    result_path = 'results/predict_' + classifier + '_' + str(N) + '-' + str(K) + '.json'

    t1 = time.time()
    model = FeatureExtractor(ckpt_path)
    model.cuda()
    extractor_feature(base_data_path, model, base_feature_path)
    extractor_feature_test(test_data_path, model, test_feature_path)
    t2 = time.time()
    print('End of feature extraction!')
    print('time: {:.2f}'.format(t2 - t1))

    t1 = time.time()
    print(f'starting {N}-way {K}-shot...')
    predict = test(base_feature_path=base_feature_path, test_feature_path=test_feature_path, result_path=result_path,
                N=N, K=K, sampled_num=sampled_num, k=k, alpha=alpha, p=p,
                classifiers=classifier)
    t2 = time.time()
    print('#################END################')
    print('see {} for prediction'.format(result_path))
    print('total time: {:.2f}'.format(t2 - t1))    


if __name__ == '__main__':
    main()
       

    
