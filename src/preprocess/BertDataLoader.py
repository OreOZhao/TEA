from collections import Counter
from itertools import chain

from transformers import BertTokenizer

from config.KBConfig import *
from tools import FileTools
from tools.Announce import Announce
from tools.MultiprocessingTool import MPTool


class BertDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        datas = self.load_data(BertDataLoader.line_to_feature)  # tokenize, return eid, tokens, tids
        self.save_data(datas)
        self.save_token_freq(datas)

    def my_run(self):
        self.load_data_and_save(BertDataLoader.line_to_feature, self.dataset.attr_seq_out, self.dataset.attr_tokens_out,
                                self.dataset.attr_tids_out, self.dataset.attr_token_freqs_out)
        self.load_data_and_save(BertDataLoader.line_to_feature, self.dataset.neighboronly_seq_out,
                                self.dataset.neighboronly_tokens_out,
                                self.dataset.neighboronly_tids_out, self.dataset.neighboronly_token_freqs_out)

    def load_data_and_save(self, line_solver, seq_out, tokens_out, tids_out, token_freqs_out):
        datas = self.load_data_from_seq_out(line_solver, seq_out)
        self.save_data_to_out(datas, tokens_out, tids_out)
        self.save_token_freq_to_out(datas, token_freqs_out)

    def load_data(self, line_solver):
        data_path = self.dataset.seq_out  # sequence_from_tab_
        print(Announce.doing(), 'load BertTokenizer')
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path)
        print(Announce.done())
        with open(data_path, 'r', encoding='utf-8') as rfile:
            datas = MPTool.packed_solver(line_solver, tokenizer=tokenizer).send_packs(rfile).receive_results()
        return datas  # (eid, tokens, tid_seq)

    def load_data_from_seq_out(self, line_solver, seq_out):
        data_path = seq_out  # sequence_from_tab_
        print(Announce.doing(), 'load BertTokenizer')
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path)
        print(Announce.done())
        with open(data_path, 'r', encoding='utf-8') as rfile:
            datas = MPTool.packed_solver(line_solver, tokenizer=tokenizer).send_packs(rfile).receive_results()
        return datas  # (eid, tokens, tid_seq)

    @staticmethod
    def line_to_feature(line: str, tokenizer: BertTokenizer):
        eid, text = line.strip('\n').strip().split('\t')
        tokens = tokenizer.tokenize(text)
        tid_seq = tokenizer.convert_tokens_to_ids(tokens)
        return int(eid), tokens, tid_seq

    def save_data(self, datas):
        tokens_path = self.dataset.tokens_out
        tids_path = self.dataset.tids_out
        tids = [(eid, tids) for eid, tokens, tids in datas]
        tokens = [(eid, tokens) for eid, tokens, tids in datas]
        FileTools.save_list_p(tids, tids_path)
        FileTools.save_list_p(tokens, tokens_path)
        # return tokens, tids

    def save_data_to_out(self, datas, tokens_out, tids_out):
        tokens_path = tokens_out
        tids_path = tids_out
        tids = [(eid, tids) for eid, tokens, tids in datas]
        tokens = [(eid, tokens) for eid, tokens, tids in datas]
        FileTools.save_list_p(tids, tids_path)
        FileTools.save_list_p(tokens, tokens_path)
        # return tokens, tids

    @staticmethod
    def load_saved_data(dataset):
        tids_path = dataset.tids_out
        tids = FileTools.load_list_p(tids_path)
        return tids

    @staticmethod
    def load_saved_type_seq(dataset, type):
        if type == 'attr':
            tids_path = dataset.attr_tids_out
        elif type == 'neighboronly':
            tids_path = dataset.neighboronly_tids_out
        else:
            tids_path = dataset.tids_out
        tids = FileTools.load_list_p(tids_path)
        return tids

    def save_token_freq(self, datas) -> dict:
        freq_path = self.dataset.token_freqs_out
        tokens = [tokens for eid, tokens, tids in datas]
        tids = [tids for eid, tokens, tids in datas]
        tokens = list(chain.from_iterable(tokens))
        tids = list(chain.from_iterable(tids))
        results = [(token, tid) for token, tid in zip(tokens, tids)]
        r_counter = Counter(results)
        # FileTools.save_dict(r_counter, freq_path)
        r_dict = dict(r_counter)
        r_list = sorted(r_dict.items(), key=lambda x: x[1], reverse=True)
        FileTools.save_list_p(r_list, freq_path)
        return r_dict

    def save_token_freq_to_out(self, datas, token_freqs_out) -> dict:
        freq_path = token_freqs_out
        tokens = [tokens for eid, tokens, tids in datas]
        tids = [tids for eid, tokens, tids in datas]
        tokens = list(chain.from_iterable(tokens))
        tids = list(chain.from_iterable(tids))
        results = [(token, tid) for token, tid in zip(tokens, tids)]
        r_counter = Counter(results)
        # FileTools.save_dict(r_counter, freq_path)
        r_dict = dict(r_counter)
        r_list = sorted(r_dict.items(), key=lambda x: x[1], reverse=True)
        FileTools.save_list_p(r_list, freq_path)
        return r_dict

    @staticmethod
    def load_freq(dataset):
        freq_path = dataset.token_freqs_out
        print(Announce.printMessage(), 'load:', freq_path)
        freq_list = FileTools.load_list_p(freq_path)
        freq_dict = {key: value for key, value in freq_list}
        return freq_dict
