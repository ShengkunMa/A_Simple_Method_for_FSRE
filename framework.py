import torch
import json
import numpy as np
import time
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from data_kit import get_valdata_feature, get_valdata_feature_test

def get_meancov(data_feature):
    feature_mean = {}
    feature_cov = {}
    for pid, fea in data_feature.items():
        mean = np.mean(fea, axis=0)
        cov = np.cov(fea.T) 
        feature_mean[pid] = mean
        feature_cov[pid] = cov 
    return feature_mean, feature_cov 

def get_calibrated_meancov(instance, feature_mean, feature_cov, k=10, p=0.5, alpha=0.21):
    dist = []
    means = []
    covs = []
    for relid in feature_mean:
        means.append(feature_mean[relid])
        covs.append(feature_cov[relid])
        dist.append(np.linalg.norm(instance - feature_mean[relid]))
    dist, means, covs = np.array(dist), np.array(means), np.array(covs)
    indices = np.argpartition(dist, k)[:k]
    dist, means, covs = dist[indices], means[indices], covs[indices]   
    dist_rec = 1 / dist        
    coe = dist_rec / np.sum(dist_rec)
    means = means * np.expand_dims(coe, axis=1)
    covs = covs * np.expand_dims(coe, axis=(2,1))
    calibrated_mean = np.sum(means, axis=0) * p + instance * (1 - p)
    calibrated_cov = np.sum(covs, axis=0) + alpha
    return calibrated_mean, calibrated_cov


def train(base_feature_path, novel_feature_path, 
        N=5, K=1, Q=5, num_task=1000, sampled_num=700, k=5, alpha=0.25, p=0.1, classifiers='LR'):

    with open(base_feature_path, 'r') as f:
        base_feature_dict = json.load(f)
    for key in base_feature_dict.keys():
        base_feature_dict[key] = np.array(base_feature_dict[key])    
    with open(novel_feature_path, 'r') as f:
        novel_feature_dict = json.load(f)
    for key in novel_feature_dict.keys():
        novel_feature_dict[key] = np.array(novel_feature_dict[key]) 

    base_mean, base_cov = get_meancov(base_feature_dict)
    val_data_loader = get_valdata_feature(novel_feature_dict, N, K, Q, num_task)
    
    acc_list = []
    for iter in range(num_task):
        tb = time.time()
        support_set, query_set, support_labels, labels = val_data_loader[iter]
        sampled_support = []
        sampled_support_labels = []

        for i in range(N * K):
            support = support_set[i]
            calibrated_mean, calibrated_cov = get_calibrated_meancov(support, base_mean, base_cov, k, p, alpha)
            sampled_support.append(np.random.multivariate_normal(calibrated_mean, calibrated_cov, size=sampled_num))
            sampled_support_labels.append(np.full((sampled_num), support_labels[i]))
        sampled_support = np.concatenate(sampled_support)
        sampled_support_labels = np.concatenate(sampled_support_labels)
        x_aug = np.concatenate([support_set, sampled_support])
        y_aug = np.concatenate([support_labels, sampled_support_labels])
        
        if classifiers == 'LR':
            classifier = LogisticRegression(max_iter=5000).fit(X=x_aug, y=y_aug)
        if classifiers == 'SVM':
            classifier = svm.SVC().fit(x_aug, y_aug)
        
        predicts = classifier.predict(query_set)
        acc = np.mean(predicts == labels)
        acc_list.append(acc)
        te = time.time()
        print('task: {}, acc: {:.4f}, mean_acc: {:.4f}, time: {:.2f}'.format(iter, acc, np.mean(acc_list), te - tb))

    acc_mean = np.mean(acc_list)
    return acc_mean

def test(base_feature_path, test_feature_path, result_path, 
        N=5, K=1, sampled_num=700, k=5, alpha=0.25, p=0.1, classifiers='LR'):

    with open(base_feature_path, 'r') as f:
        base_feature_dict = json.load(f)
    for key in base_feature_dict.keys():
        base_feature_dict[key] = np.array(base_feature_dict[key])    
    with open(test_feature_path, 'r') as f:
        test_feature_dict = json.load(f)
    for task in test_feature_dict:
        task['meta_test'] = np.array(task['meta_test'])
        task['meta_train'] = np.array(task['meta_train'])

    base_mean, base_cov = get_meancov(base_feature_dict)
    val_data_loader = get_valdata_feature_test(test_feature_dict, N, K)
    
    predicts_list = []
    for iter, item in enumerate(val_data_loader):
        tb = time.time()
        support_set, query_set, support_labels = item

        sampled_support = []
        sampled_support_labels = []
        for i in range(N * K):
            support = support_set[i]
            calibrated_mean, calibrated_cov = get_calibrated_meancov(support, base_mean, base_cov, k, p, alpha)
            sampled_support.append(np.random.multivariate_normal(calibrated_mean, calibrated_cov, size=sampled_num))
            sampled_support_labels.append(np.full((sampled_num), support_labels[i]))
        sampled_support = np.concatenate(sampled_support)
        sampled_support_labels = np.concatenate(sampled_support_labels)
        x_aug = np.concatenate([support_set, sampled_support])
        y_aug = np.concatenate([support_labels, sampled_support_labels])

        if classifiers == 'LR':
            classifier = LogisticRegression(max_iter=5000).fit(X=x_aug, y=y_aug)
        if classifiers == 'SVM':
            classifier = svm.SVC().fit(x_aug, y_aug)
        
        predicts = classifier.predict(query_set)
        predicts = predicts.tolist()
        predicts_list += predicts        
        te = time.time()
        print('task: {}, pred: {:.4f}, time: {:.2f}'.format(iter, predicts, te - tb))

    if not os.path.exists('./result'):
        os.mkdir('./result')
    with open(result_path, 'w') as f:
        json.dump(predicts_list, f)
    return predicts_list



