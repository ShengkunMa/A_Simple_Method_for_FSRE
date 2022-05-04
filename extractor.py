import torch
import torch.nn as nn
from transformers import BertModel


class FeatureExtractor(nn.Module):    
    def __init__(self, path): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.load_state_dict(torch.load(path, map_location='cpu')["bert-base"], False)
        print("We load "+ path +" for bert!")

    def forward(self, inputs):
        outputs = self.bert(inputs['ids'], attention_mask=inputs['mask'])
        tensor_range = torch.arange(inputs['ids'].size()[0])
        h_state = outputs[0][tensor_range, inputs["pos1"]]
        t_state = outputs[0][tensor_range, inputs["pos2"]]
        state = torch.cat((h_state, t_state), -1) 
        return state



        