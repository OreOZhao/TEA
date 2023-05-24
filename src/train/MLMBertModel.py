from transformers import BertModel
import torch as t
from config.KBConfig import *
from tools.Announce import Announce


class MLMBertModel(t.nn.Module):
    def __init__(self, pretrain_bert_path, prompt_labels_tids):
        super(MLMBertModel, self).__init__()
        bert_config = BertConfig.from_pretrained(pretrain_bert_path)
        self.num_labels = num_prompt_labels
        self.label_list = prompt_labels_tids
        self.bert_model = BertModel.from_pretrained(pretrain_bert_path, config=bert_config)
        self.bert_model.resize_token_embeddings(len_tokenizer)
        self.out_linear_layer = t.nn.Linear(bert_config.hidden_size, bert_output_dim)
        self.prompt_linear_layer = t.nn.Linear(bert_config.hidden_size, bert_config.vocab_size)
        self.dropout = t.nn.Dropout(p=0.1)
        print(Announce.printMessage(), '--------Init MLMBertModel--------')

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

    def get_mask_output(self, tids, mask_prompt):
        bs = tids.shape[0]
        bert_out = self.bert_model(input_ids=tids, attention_mask=mask_prompt)
        # bert_out = self.bert_model(input_ids=tids, attention_mask=mask_prompt, output_attentions=True) # visualization
        # layer * (bs, num_heads, seq_len, seq_len)
        sequence_output = bert_out.last_hidden_state
        mask_idx = (tids == 103).nonzero(as_tuple=True)[1]
        mask_output = sequence_output[t.arange(bs), mask_idx]
        mask_output = self.dropout(mask_output)
        prediction_mask_scores = self.prompt_linear_layer(mask_output)
        logits = []
        for label_tid in self.label_list:
            logits.append(prediction_mask_scores[:, label_tid].unsqueeze(-1))
        logits = t.cat(logits, -1)
        return logits

    def forward(self, tids, masks):
        mask0 = masks[:, 0]  # prompt
        mask1 = masks[:, 1]  # cls 1
        mask2 = masks[:, 2]  # cls 2
        prompt_logits = self.get_mask_output(tids, mask0)
        cls1 = self.get_cls_output(tids, mask1)
        cls2 = self.get_cls2_output(tids, mask2)
        return prompt_logits, cls1, cls2
