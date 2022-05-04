import json
import torch
import time
import random
import os
import numpy as np
from transformers import BertTokenizer

def get_valdata_feature(feature_dict, N, K, Q, num_task): 
    relids = list(feature_dict.keys())
    data_loader = []
    for iter in range(num_task):
        targets = random.sample(relids, N)
        support_feature = []
        query_feature = []
        support_labels = []
        labels = []
        for i, relid in enumerate(targets):
            indices = np.random.choice(list(range(feature_dict[relid].shape[0])), K + Q, False)
            count = 0
            for j in indices:
                ins_fea = feature_dict[relid][j]
                if count < K:
                    support_feature.append(ins_fea)
                else:
                    query_feature.append(ins_fea)
                count += 1
            support_labels += [i] * K
            labels += [i] * Q 
        support_feature = np.stack(support_feature, 0)
        query_feature = np.stack(query_feature, 0)
        labels = np.array(labels)
        support_labels = np.array(support_labels)
        data_loader.append([support_feature, query_feature, support_labels, labels])
    return data_loader

def get_valdata_feature_test(test_feature, N, K): 
    data_loader = []
    for task in test_feature:
        support_labels = []
        for i in range(N):
            support_labels += [i] * K
        support_labels = np.array(support_labels)
        data_loader.append([task['meta_train'], task['meta_test'], support_labels])
    return data_loader

def extractor_feature(data_path, model, save_path, batch_size=4, max_length=128):
    t1 = time.time()
    data_loader = get_dataloader(data_path, batch_size, max_length)
    t2 = time.time()
    print('data_loader time: {:.2f}'.format(t2 - t1))
    data_feature = {} 
    t3 = time.time()
    with torch.no_grad():
        for pid, datas in data_loader.items():
            fea = []
            batch_num = datas['ids'].size()[0]
            for i in range(batch_num):
                inputs = {}
                inputs['ids'], inputs['pos1'], inputs['pos2'], inputs['mask'] = datas['ids'][i], datas['pos1'][i], datas['pos2'][i], datas['mask'][i]
                inputs['ids'], inputs['pos1'], inputs['pos2'], inputs['mask'] = inputs['ids'].cuda(), inputs['pos1'].cuda(), inputs['pos2'].cuda(), inputs['mask'].cuda()
                outputs = model(inputs)
                batch_features = outputs.cpu().data.numpy()   
                for i in range(batch_size):
                    fea.append(batch_features[i])  
            fea = np.stack(fea, 0)
            data_feature[pid] = fea
    t4 = time.time()
    print('model computing time: {:.2f}'.format(t4 - t3))
    for pid in data_feature:
        data_feature[pid] = data_feature[pid].tolist()
    if not os.path.exists('./feature'):
        os.mkdir('./feature')
    with open(save_path, 'w') as f:
        json.dump(data_feature, f)

def extractor_feature_test(data_path, model, save_path, max_length=128):
    t1 = time.time()
    data_loader = get_dataloader_test(data_path, max_length)
    t2 = time.time()
    print(f'data_loader time: {t2 - t1}')
    data_feature = []
    with torch.no_grad():
        for task in data_loader:
            feature_meta = {}
            feature_meta['relation'] = task['relation']
            for k in task['meta_test']:
                task['meta_test'][k] = task['meta_test'][k].cuda()
                task['meta_train'][k] = task['meta_train'][k].cuda()
            outputs_test = model(task['meta_test'])
            outputs_train = model(task['meta_train'])
            test_feature, train_feature = outputs_test[1].cpu().data.numpy().tolist(), outputs_train[1].cpu().data.numpy().tolist()
            feature_meta['meta_test'] = test_feature
            feature_meta['meta_train'] = train_feature
            data_feature.append(feature_meta)
    t3 = time.time()
    print(f'model time: {t3 - t2}')
    if not os.path.exists('./feature'):
        os.mkdir('./feature')
    with open(save_path, 'w') as f:
        json.dump(data_feature, f)

def get_dataloader(data_path, batch_size, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(data_path, 'r') as f:
        json_data = json.load(f)
    relids = list(json_data.keys())
    data_loader = {}
    for relid in relids:
        raw_data = json_data[relid]
        batch_num = int(len(raw_data) / batch_size)
        data_batch = {'ids': [], 'pos1': [], 'pos2': [], 'mask': []}
        for i in range(batch_num):
            abatch = {'ids': [], 'pos1': [], 'pos2': [], 'mask': []}
            for j in range(batch_size):
                instance = raw_data[i * batch_size + j]
                raw_tokens = instance['tokens']
                head = instance['h'][2][0]
                tail = instance['t'][2][0]
                ids, pos1, pos2, mask = tokenize(raw_tokens, head, tail, tokenizer, max_length)
                ids, pos1, pos2, mask = torch.tensor(ids).long(), torch.tensor(pos1).long(), torch.tensor(pos2).long(), torch.tensor(mask).long()
                abatch['ids'].append(ids)
                abatch['pos1'].append(pos1)
                abatch['pos2'].append(pos2)
                abatch['mask'].append(mask)
            for k in abatch:
                data_batch[k].append(torch.stack(abatch[k], 0))
        for k in data_batch:
            data_batch[k] = torch.stack(data_batch[k], 0)
        data_loader[relid] = data_batch
    return data_loader 

def get_dataloader_test(data_path, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(data_path, 'r') as f:
        json_data = json.load(f)
    data_loader = []
    for task in json_data:
        task_loader = {}
        task_loader['relation'] = task['relation']
        meta_test = {'ids': [], 'pos1': [], 'pos2': [], 'mask': []}
        meta_train = {'ids': [], 'pos1': [], 'pos2': [], 'mask': []}

        raw_tokens = task['meta_test']['tokens']
        head = task['meta_test']['h'][2][0]
        tail = task['meta_test']['t'][2][0]
        ids, pos1, pos2, mask = tokenize(raw_tokens, head, tail, tokenizer, max_length)
        ids, pos1, pos2, mask = torch.tensor(ids).long(), torch.tensor(pos1).long(), torch.tensor(pos2).long(), torch.tensor(mask).long()
        meta_test['ids'].append(ids)
        meta_test['pos1'].append(pos1)
        meta_test['pos2'].append(pos2)
        meta_test['mask'].append(mask)
        for k in meta_test:
            meta_test[k] = torch.stack(meta_test[k], 0)
        task_loader['meta_test'] = meta_test

        N = len(task['meta_train'])
        K = len(task['meta_train'][0])
        for n in range(N):
            for k in range(K):
                raw_tokens = task['meta_train'][n][k]['tokens']
                head = task['meta_train'][n][k]['h'][2][0]
                tail = task['meta_train'][n][k]['t'][2][0]
                ids, pos1, pos2, mask = tokenize(raw_tokens, head, tail, tokenizer, max_length)
                ids, pos1, pos2, mask = torch.tensor(ids).long(), torch.tensor(pos1).long(), torch.tensor(pos2).long(), torch.tensor(mask).long()
                meta_train['ids'].append(ids)
                meta_train['pos1'].append(pos1)
                meta_train['pos2'].append(pos2)
                meta_train['mask'].append(mask)
        for k in meta_train:
            meta_train[k] = torch.stack(meta_train[k], 0)       
        task_loader['meta_train'] = meta_train
        data_loader.append(task_loader)
    return data_loader

def tokenize(raw_tokens, pos_head, pos_tail, tokenizer, max_length=128):
    # token -> index
    tokens = ['[CLS]']
    cur_pos = 0
    pos1_in_index = 0
    pos2_in_index = 0
    for token in raw_tokens:
        token = token.lower()
        if cur_pos == pos_head[0]:
            tokens.append('[unused0]')
            pos1_in_index = len(tokens)
        if cur_pos == pos_tail[0]:
            tokens.append('[unused1]')
            pos2_in_index = len(tokens)
        tokens += tokenizer.tokenize(token)
        if cur_pos == pos_head[-1]:
            tokens.append('[unused2]')
        if cur_pos == pos_tail[-1]:
            tokens.append('[unused3]')
        cur_pos += 1

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    
    # padding
    while len(indexed_tokens) < max_length:
        indexed_tokens.append(0)
    indexed_tokens = indexed_tokens[:max_length]

    # pos
    pos1 = np.zeros((max_length), dtype=np.int32)
    pos2 = np.zeros((max_length), dtype=np.int32)
    for i in range(max_length):
        pos1[i] = i - pos1_in_index + max_length
        pos2[i] = i - pos2_in_index + max_length

    # mask
    mask = np.zeros((max_length), dtype=np.int32)
    mask[:len(tokens)] = 1
    
    if pos1_in_index == 0:
        pos1_in_index = 1
    if pos2_in_index == 0:
        pos2_in_index = 1
    pos1_in_index = min(max_length, pos1_in_index)
    pos2_in_index = min(max_length, pos2_in_index)

    return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask
