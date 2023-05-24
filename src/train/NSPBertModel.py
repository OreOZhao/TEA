from transformers import BertModel
import torch as t
from config.KBConfig import *
from tools.Announce import Announce
import torch.nn as nn
import torch.nn.functional as F


class NSPBertModel(t.nn.Module):
    def __init__(self, pretrain_bert_path, prompt_labels_tids):
        super(NSPBertModel, self).__init__()
        bert_config = BertConfig.from_pretrained(pretrain_bert_path)
        self.bert_model = BertModel.from_pretrained(pretrain_bert_path, config=bert_config)
        self.bert_model.resize_token_embeddings(len_tokenizer)
        self.out_linear_layer = t.nn.Linear(bert_config.hidden_size, bert_output_dim)
        self.nsp_linear_layer = t.nn.Linear(bert_config.hidden_size, 2)
        self.dropout = t.nn.Dropout(p=0.1)
        print(Announce.printMessage(), '--------Init NSPBertModel--------')

    def get_cls_output(self, tids, mask_cls):
        bert_out = self.bert_model(input_ids=tids, attention_mask=mask_cls)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 0]
        output = self.dropout(cls)
        output = self.out_linear_layer(output)
        return output

    def get_cls2_output(self, tids, mask_cls):
        bert_out = self.bert_model(input_ids=tids, attention_mask=mask_cls)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 1]
        output = self.dropout(cls)
        output = self.out_linear_layer(output)
        return output

    def get_NSP_output(self, tids, mask_prompt):
        bs = tids.shape[0]
        bert_out = self.bert_model(input_ids=tids, attention_mask=mask_prompt)
        last_hidden_state = bert_out.last_hidden_state
        cls = last_hidden_state[:, 0]
        nsp_output = self.dropout(cls)
        nsp_output = self.nsp_linear_layer(nsp_output)
        return nsp_output

    def forward(self, tids, masks):
        mask0 = masks[:, 0]  # prompt
        mask1 = masks[:, 1]  # cls 1
        mask2 = masks[:, 2]  # cls 2
        nsp_logits = self.get_NSP_output(tids, mask0)
        cls1 = self.get_cls_output(tids, mask1)
        cls2 = self.get_cls2_output(tids, mask2)
        return nsp_logits, cls1, cls2
