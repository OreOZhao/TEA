import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertTokenizer, BertModel, AdamW

from config.KBConfig import *
from preprocess import Parser
from preprocess.KBStore import KBStore
from tools.Announce import Announce
from tools.ModelTools import ModelTools
from tools.MultiprocessingTool import MPTool, MultiprocessingTool
from tools.TrainingTools import TrainingTools
from train.PairwisePromptDataset import PairwisePromptDataset, PairwisePromptTestDataset
from train.MLMBertModel import MLMBertModel
from train.NSPBertModel import NSPBertModel
from train.train_utils import cos_sim_mat_generate, batch_topk, hits
import pickle

VALID = True


class PairwisePromptTrainer(MultiprocessingTool):
    def __init__(self):
        super(PairwisePromptTrainer, self).__init__()
        self.get_emb_batch = 512
        self.train_emb_batch = 8
        # self.emb_cos_batch = 512
        self.nearest_sample_num = 128
        self.train_cands = 50
        self.reranking_cands = 128
        self.reranking_thres_prob = 0.9
        self.neg_num = 2
        self.score_distance_level = SCORE_DISTANCE_LEVEL
        self.neighbor = False
        print(Announce.printMessage(), 'reranking_cands', self.reranking_cands)
        print(Announce.printMessage(), 'reranking_thres', self.reranking_thres_prob)
        print(Announce.printMessage(), '--------Init PairwisePromptTrainer--------')

    def data_prepare(
            self, eid2tids1: dict, eid2tids2: dict,
            fs1: KBStore, fs2: KBStore,
            eid2nei_tids1=None, eid2nei_tids2=None):
        self.eid2tids1 = eid2tids1
        self.eid2tids2 = eid2tids2
        if args.neighbor and eid2nei_tids1 is not None and eid2nei_tids2 is not None:
            self.neighbor = True
            self.eid2nei_tids1 = eid2nei_tids1
            self.eid2nei_tids2 = eid2nei_tids2
        self.fs1 = fs1
        self.fs2 = fs2
        tokenizer = BertTokenizer.from_pretrained(args.pretrain_bert_path)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        self.cls_token, self.sep_token = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        self.template = template

        print(Announce.printMessage(), "template: ", self.template)
        self.temp_tids = tokenizer(self.template)['input_ids'][1:-1]
        self.eid2name_tids1 = None
        self.eid2name_tids2 = None
        self.prompt_labels_tids = []
        for t in prompt_labels:
            id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))
            self.prompt_labels_tids.append(id[0])

        eid2data1 = [self.tids_solver(item, cls_token=self.cls_token, sep_token=self.sep_token, freqs=None) for item in
                     eid2tids1.items()]  # tid_solver returns: eid, (input_ids, masks)
        self.eid2data1 = {key: value for key, value in eid2data1}

        eid2data2 = [self.tids_solver(item, cls_token=self.cls_token, sep_token=self.sep_token, freqs=None) for item in
                     eid2tids2.items()]
        self.eid2data2 = {key: value for key, value in eid2data2}
        print(Announce.printMessage(), 'eid2data1 len:', len(self.eid2data1))
        print(Announce.printMessage(), 'eid2data2 len:', len(self.eid2data2))

        if not args.blocking:
            # 读取 train_links, valid_links
            self.train_links = self.load_links(links.train, fs1.entity_ids, fs2.entity_ids)
            self.valid_links = self.load_links(links.valid, fs1.entity_ids, fs2.entity_ids)
            self.test_links = self.load_links(links.test, fs1.entity_ids, fs2.entity_ids)
            self.train_links = list(set(self.train_links))
            self.valid_links = list(set(self.valid_links))
            self.test_links = list(set(self.test_links))
            print(Announce.printMessage(), 'train links len:', len(self.train_links))
            print(Announce.printMessage(), 'valid links len:', len(self.valid_links))
            print(Announce.printMessage(), 'test links len:', len(self.test_links))
            self.train_links_p = [(e1, e2) for e1, e2 in self.train_links if
                                  e1 in self.eid2data1 and e2 in self.eid2data2]
            self.valid_links_p = [(e1, e2) for e1, e2 in self.valid_links if
                                  e1 in self.eid2data1 and e2 in self.eid2data2]
            self.test_links_p = [(e1, e2) for e1, e2 in self.test_links if
                                 e1 in self.eid2data1 and e2 in self.eid2data2]
            self.train_ent1s = list({e1 for e1, e2 in self.train_links})
            self.train_ent2s = list({e2 for e1, e2 in self.train_links})
            self.valid_ent1s = [e1 for e1, e2 in self.valid_links]
            self.valid_ent2s = [e2 for e1, e2 in self.valid_links]
            self.test_ent1s = [e1 for e1, e2 in self.test_links]
            self.test_ent2s = [e2 for e1, e2 in self.test_links]

            self.train_ent1s_p = list({e1 for e1, e2 in self.train_links_p})
            self.train_ent2s_p = list({e2 for e1, e2 in self.train_links_p})
            self.valid_ent1s_p = [e1 for e1, e2 in self.valid_links_p]
            self.valid_ent2s_p = [e2 for e1, e2 in self.valid_links_p]
            self.test_ent1s_p = [e1 for e1, e2 in self.test_links_p]
            self.test_ent2s_p = [e2 for e1, e2 in self.test_links_p]
            self.all_ent1s_p = list(self.eid2data1.keys())  # eids
            self.all_ent2s_p = list(self.eid2data2.keys())
            self.all_ent1s = list(fs1.entity_ids.values())  # (input_ids, masks)
            self.all_ent2s = list(fs2.entity_ids.values())
            self.all_ent1s_p_idx = [self.all_ent1s.index(ent) for ent in self.all_ent1s_p]
            self.all_ent2s_p_idx = [self.all_ent2s.index(ent) for ent in self.all_ent2s_p]

            self.block_loader1, self.block_loader2 = self.links_pair_loader(self.all_ent1s_p, self.all_ent2s_p)
            if VALID:
                self.valid_link_loader1, self.valid_link_loader2 = self.links_pair_loader(self.valid_ent1s_p,
                                                                                          self.valid_ent2s_p)
            self.test_link_loader1, self.test_link_loader2 = self.links_pair_loader(self.test_ent1s_p,
                                                                                    self.test_ent2s_p)

        if self.neighbor:
            eid2nei_data1 = [self.tids_solver(item, cls_token=self.cls_token, sep_token=self.sep_token, freqs=None) for
                             item in
                             eid2nei_tids1.items()]  # tid_solver returns: eid, (input_ids, masks)
            self.eid2nei_data1 = {key: value for key, value in eid2nei_data1}

            eid2nei_data2 = [self.tids_solver(item, cls_token=self.cls_token, sep_token=self.sep_token, freqs=None) for
                             item in
                             eid2nei_tids2.items()]  # tid_solver returns: eid, (input_ids, masks)
            self.eid2nei_data2 = {key: value for key, value in eid2nei_data2}

            self.nei_test_link_loader1, self.nei_test_link_loader2 = self.links_pair_data_loader(self.test_ent1s_p,
                                                                                                 self.test_ent2s_p,
                                                                                                 self.eid2nei_data1,
                                                                                                 self.eid2nei_data2)

    def links_pair_loader(self, ent1s, ent2s):
        inputs1 = self.get_tensor_data(ent1s, self.eid2data1)
        inputs2 = self.get_tensor_data(ent2s, self.eid2data2)
        ds1 = TensorDataset(*inputs1)
        ds2 = TensorDataset(*inputs2)
        sampler1 = SequentialSampler(ds1)
        sampler2 = SequentialSampler(ds2)
        loader1 = DataLoader(ds1, sampler=sampler1, batch_size=self.get_emb_batch)
        loader2 = DataLoader(ds2, sampler=sampler2, batch_size=self.get_emb_batch)
        return loader1, loader2

    def links_pair_data_loader(self, ent1s, ent2s, eid2data1, eid2data2):
        inputs1 = self.get_tensor_data(ent1s, eid2data1)
        inputs2 = self.get_tensor_data(ent2s, eid2data2)
        ds1 = TensorDataset(*inputs1)
        ds2 = TensorDataset(*inputs2)
        sampler1 = SequentialSampler(ds1)
        sampler2 = SequentialSampler(ds2)
        loader1 = DataLoader(ds1, sampler=sampler1, batch_size=self.get_emb_batch)
        loader2 = DataLoader(ds2, sampler=sampler2, batch_size=self.get_emb_batch)
        return loader1, loader2

    @staticmethod
    def get_tensor_data(ents: list, eid2data: dict):
        inputs = [eid2data.get(key) for key in ents]
        print(len(inputs))
        input_ids = [ids for ids, mask in inputs]
        input_ids = t.stack(input_ids, dim=0)
        masks = [mask for ids, mask in inputs]
        masks = t.stack(masks, dim=0)
        return input_ids, masks

    def train(
            self, epochs=100,
            max_grad_norm=1.0,
            learning_rate=2e-5,
            adam_eps=1e-8,
            warmup_steps=0,
            weight_decay=0.0,
            device='cpu',
    ):
        if args.basic_bert_path is not None:
            print(Announce.printMessage(), 'Loading model from ' + args.basic_bert_path)
            bert_model = ModelTools.load_model(args.basic_bert_path)
        else:
            if args.nsp:
                bert_model = NSPBertModel(args.pretrain_bert_path, self.prompt_labels_tids)
            else:
                bert_model = MLMBertModel(args.pretrain_bert_path, self.prompt_labels_tids)

        if PARALLEL:
            bert_model = t.nn.DataParallel(bert_model)
        bert_model.to(device)
        criterion_margin = t.nn.MarginRankingLoss(MARGIN)
        criterion_ce = t.nn.CrossEntropyLoss()
        criterion_prompt_margin = t.nn.MarginRankingLoss(PROMPT_MARGIN)
        optimizer = AdamW(bert_model.parameters(), lr=1e-5)

        # eval reranking then exit
        # if args.basic_bert_path:
        #     self.reranking_cands = args.reranking_cands
        #     self.reranking_thres_prob = args.reranking_thres
        #     print(Announce.printMessage(), 'reranking_cands', self.reranking_cands)
        #     print(Announce.printMessage(), 'reranking_thres', self.reranking_thres_prob)
        #     attr_hits = self.get_hits(bert_model, self.test_link_loader1, self.test_link_loader2, device=device)
        #     print("attr hits: ", attr_hits)
        #     if self.neighbor:
        #         nei_hits = self.get_hits(bert_model, self.nei_test_link_loader1, self.nei_test_link_loader2,
        #                                  device=device)
        #         print("neighbor hits: ", nei_hits)
        #     if self.neighbor:
        #         if attr_hits[0] < nei_hits[0]:
        #             print(Announce.printMessage(), "Re-ranking with neighbor")
        #             self.get_hits(bert_model, self.nei_test_link_loader1, self.nei_test_link_loader2,
        #                           device=device, rerank=True)
        #         else:
        #             print(Announce.printMessage(), "Re-ranking with attribute")
        #             self.get_hits(bert_model, self.test_link_loader1, self.test_link_loader2, device=device,
        #                           rerank=True)
        #     exit()

        if VALID:
            model_tool = ModelTools(3, 'max')
        for epoch in range(1, epochs + 1):
            print(Announce.doing(), 'Epoch', epoch, '/', epochs, 'start')

            train_tups = self.generate_train_tups(bert_model, self.train_ent1s_p, self.train_ent2s_p,
                                                  self.train_links_p, device)  # [pe1, pe2, ne1, ne2] # no grad
            # ne1, ne2 = random([pe1, ne2] or [ne1, pe2])
            print(Announce.printMessage(), 'train_tups.shape:', np.array(train_tups).shape)

            bert_model.train()  # with grad
            train_ds = PairwisePromptDataset(train_tups, self.eid2tids1, self.eid2tids2, self.temp_tids,
                                             self.cls_token, self.sep_token,
                                             self.eid2name_tids1, self.eid2name_tids2)

            train_samp = SequentialSampler(train_ds)
            train_loader = DataLoader(train_ds, sampler=train_samp, batch_size=self.train_emb_batch)
            tt = TrainingTools(train_loader, device=device)

            if self.neighbor:
                nei_train_ds = PairwisePromptDataset(train_tups, self.eid2nei_tids1, self.eid2nei_tids2,
                                                     self.temp_tids, self.cls_token, self.sep_token,
                                                     self.eid2name_tids1, self.eid2name_tids2)
                nei_train_samp = SequentialSampler(nei_train_ds)
                nei_train_loader = DataLoader(nei_train_ds, sampler=nei_train_samp, batch_size=self.train_emb_batch)
                # batch: self.ent2data1[pe1], self.ent2data2[pe2], self.ent2data1[ne1], self.ent2data2[ne2]
                nei_tt = TrainingTools(nei_train_loader, device=device)
                for batch1, batch2 in zip(tt.batches(lambda batch: len(batch[0][0])),
                                          nei_tt.batches(lambda batch: len(batch[0][0]), type='neighbor')):
                    i, batch1 = batch1
                    optimizer.zero_grad()
                    loss = self.train_batch_prompt(bert_model, tt, batch1, criterion_margin, criterion_ce,
                                                   criterion_prompt_margin, device)

                    loss.backward()
                    optimizer.step()

                    i, batch2 = batch2
                    optimizer.zero_grad()
                    loss = self.train_batch_prompt(bert_model, nei_tt, batch2, criterion_margin, criterion_ce,
                                                   criterion_prompt_margin, device)

                    loss.backward()
                    optimizer.step()
            else:
                for i, batch in tt.batches(lambda batch: len(batch[0][0])):
                    optimizer.zero_grad()
                    loss = self.train_batch_prompt(bert_model, tt, batch, criterion_margin, criterion_ce,
                                                   criterion_prompt_margin, device)

                    loss.backward()
                    optimizer.step()
                print()

            # print()
            if VALID:
                print('valid')
                bert_model.eval()
                with t.no_grad():
                    hit_values = self.get_hits(bert_model, self.valid_link_loader1, self.valid_link_loader2,
                                               device=device)
                    stop = model_tool.early_stopping(bert_model, links.model_save, hit_values[0])
            else:
                ModelTools.save_model(bert_model, links.model_save)
            print(Announce.printMessage(), 'test phase')
            attr_hits = self.get_hits(bert_model, self.test_link_loader1, self.test_link_loader2, device=device)
            print("attr hits: ", attr_hits)
            if self.neighbor:
                nei_hits = self.get_hits(bert_model, self.nei_test_link_loader1, self.nei_test_link_loader2,
                                         device=device)
                print("neighbor hits: ", nei_hits)
            print(Announce.done(), 'Epoch', epoch, '/', epochs, 'end')
            if VALID:
                if epoch > 5 and stop:
                    print(Announce.done(), 'Training Early Stopped')
                    print(Announce.done(), 'RERANKING')
                    bert_model = model_tool.load_model(links.model_save)
                    print(Announce.printMessage(), 'Loading model from ' + links.model_save)
                    if PARALLEL:
                        bert_model = t.nn.DataParallel(bert_model)
                    bert_model.to(device)
                    if self.neighbor:
                        if attr_hits[0] < nei_hits[0]:
                            print(Announce.printMessage(), "Re-ranking with neighbor")
                            self.get_hits(bert_model, self.nei_test_link_loader1, self.nei_test_link_loader2,
                                          device=device, rerank=True)
                        else:
                            print(Announce.printMessage(), "Re-ranking with attribute")
                            self.get_hits(bert_model, self.test_link_loader1, self.test_link_loader2, device=device,
                                          rerank=True)
                    stop = False
                    break
            pass
        pass

    def get_prompt_re_rank(self, bert_model, test_links, candidates, device, tids1, tids2):
        test_tups = test_links
        # topk_idx [Ents, Candidates]
        ent_candidates = [t.tensor(self.test_ent2s)[candidates[i]].unsqueeze(0) for i in range(len(candidates))]
        ent_candidates = t.cat(ent_candidates)
        test_ds = PairwisePromptTestDataset(test_tups, ent_candidates, tids1, tids2, self.temp_tids,
                                            self.cls_token, self.sep_token, train=False,
                                            ent2name1=self.eid2name_tids1, ent2name2=self.eid2name_tids2)

        test_samp = SequentialSampler(test_ds)
        test_loader = DataLoader(test_ds, sampler=test_samp, batch_size=self.train_emb_batch)
        # logits = [prob_0, prob_1], prob_1 == entity similarity
        logits = []
        with t.no_grad():
            bert_model.eval()
            for i, batch in TrainingTools.batch_iter(test_loader, 'get prompt output'):
                seq_tids, seq_masks, seq_tids_r, seq_masks_r = batch  # [C, seq_len] [C, 3, seq_len]
                bs, cands, mask_cnt, seq_len = seq_masks.shape
                seq_tids = seq_tids.view(bs * cands, -1)
                seq_masks = seq_masks.view(bs * cands, mask_cnt, -1)
                seq_tids_r = seq_tids_r.view(bs * cands, -1)
                seq_masks_r = seq_masks_r.view(bs * cands, mask_cnt, -1)
                # [bs, cands, seq_len], [bs, cands, 3, seq_len]
                my_logits = []
                for i in range(0, bs * cands, bs):
                    pos_logits, pos_emb1, pos_emb2 = bert_model(seq_tids[i:min(i + bs, bs * cands)].to(device),
                                                                seq_masks[i:min(i + bs, bs * cands)].to(device))
                    res = pos_logits[:, 1]
                    my_logits.append(res)
                my_logits = t.cat(my_logits).view(bs, cands)  # bs * cands
                logits.append(my_logits)
        logits = t.cat(logits)  # E * cands
        return logits

    def get_hits(self, bert_model: t.nn.Module, link_loader1, link_loader2, device, rerank=False):
        # link_loader load [tids, masks]
        valid_emb1s = self.get_emb_valid(link_loader1, bert_model, device=device)
        valid_emb2s = self.get_emb_valid(link_loader2, bert_model, device=device)
        # get cls embeddings from get_emb_valid in batch
        print(Announce.printMessage(), 'valid_emb1s.shape:', valid_emb1s.shape)
        print(Announce.printMessage(), 'valid_emb2s.shape:', valid_emb2s.shape)
        cos_sim_mat = cos_sim_mat_generate(valid_emb1s, valid_emb2s, device=device)  # [Ent, Ent]
        if not rerank:
            top_scores, topk_idx = batch_topk(cos_sim_mat, topn=self.nearest_sample_num, largest=True)  # [10500, 128]
            return hits(topk_idx)
        else:
            top_scores, topk_idx = batch_topk(cos_sim_mat, topn=self.reranking_cands, largest=True)  # [10500, 128]
            print("without rerank hits: ", hits(topk_idx))
            print(Announce.printMessage(), 'START RERANKING')
            rerank_idx = top_scores[:, 0] < self.reranking_thres_prob
            rerank_test_links = [self.test_links[i] for i in range(len(rerank_idx)) if rerank_idx[i]]

            topk_idx_rerank = [topk_idx[i] for i in range(len(rerank_idx)) if rerank_idx[i]]
            print(Announce.printMessage(), 'Reranking')
            attr_prompt_scores = self.get_prompt_re_rank(bert_model, rerank_test_links, topk_idx_rerank, device,
                                                         self.eid2tids1, self.eid2tids2)
            rerank_cands = [topk_idx_rerank[i][attr_prompt_scores[i].topk(self.reranking_cands)[1]]
                            for i in range(len(topk_idx_rerank))]
            final_topk_idx = []
            cnt = 0
            for i in range(len(topk_idx)):
                if rerank_idx[i]:
                    final_topk_idx.append(rerank_cands[cnt])
                    cnt += 1
                else:
                    final_topk_idx.append(topk_idx[i])
            final_topk_idx = t.cat(final_topk_idx).view(len(topk_idx), -1)
            print("after rerank hits: ", hits(final_topk_idx))
            return hits(final_topk_idx)

    def train_batch_prompt(self, bert_model, tt, batch, criterion_margin, criterion_ce, criterion_prompt_margin,
                           device):
        # batch: seq[p1,p2], seq[n1, n2]
        # tids, [mask1,m2,m3]
        pos_logits, pos_emb1, pos_emb2 = bert_model(batch[0][0].to(device), batch[0][1].to(device))
        pos_logits_reverse, pos_emb1_reverse, pos_emb2_reverse = bert_model(batch[1][0].to(device),
                                                                            batch[1][1].to(device))
        neg_logits, neg_emb1, neg_emb2 = bert_model(batch[2][0].to(device), batch[2][1].to(device))
        neg_logits_reverse, neg_emb1_reverse, neg_emb2_reverse = bert_model(batch[3][0].to(device),
                                                                            batch[3][1].to(device))
        batch_size = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1, pos_emb2, p=self.score_distance_level, keepdim=True)
        pos_score_reverse = F.pairwise_distance(pos_emb1_reverse, pos_emb2_reverse, p=self.score_distance_level,
                                                keepdim=True)

        neg_score = F.pairwise_distance(neg_emb1, neg_emb2, p=self.score_distance_level, keepdim=True)
        neg_score_reverse = F.pairwise_distance(neg_emb1_reverse, neg_emb2_reverse, p=self.score_distance_level,
                                                keepdim=True)

        labels = t.cat([t.ones([batch_size], dtype=t.long), t.zeros([batch_size], dtype=t.long)]).to(device)
        y_pred = t.cat([pos_logits, neg_logits])  # pred from t.cosine_similarity
        y_pred_reverse = t.cat([pos_logits_reverse, neg_logits_reverse])

        loss_ce = (criterion_ce(y_pred, labels) + criterion_ce(y_pred_reverse, labels)) / 2  # cross entropy loss

        y = -t.ones(pos_score.shape).to(device)  # pos_score < neg_score, (distance)
        loss_margin = (criterion_margin(pos_score, neg_score, y)
                       + criterion_margin(pos_score_reverse, neg_score_reverse, y)) / 2

        y = t.ones(pos_score.shape).to(device)  # pos_yes > neg_yes
        loss_prompt_margin = (criterion_prompt_margin(pos_logits[:, 1].unsqueeze(-1),
                                                      neg_logits[:, 1].unsqueeze(-1), y)
                              + criterion_prompt_margin(pos_logits_reverse[:, 1].unsqueeze(-1),
                                                        neg_logits_reverse[:, 1].unsqueeze(-1), y)) / 2

        if PARALLEL:
            loss_margin = loss_margin.mean()
            loss_ce = loss_ce.mean()
            loss_prompt_margin = loss_prompt_margin.mean()
        print("%.3f" % float(loss_margin), end='\t')
        print("%.3f" % float(loss_ce), end='\t')
        print("%.3f" % float(loss_prompt_margin), end='\t')
        loss = args.lambda_prompt_ce * loss_ce + args.lambda_prompt_margin * loss_prompt_margin + loss_margin
        tt.update_metrics(loss, y_pred, labels,
                          batch_size=batch_size * 2)  # update binary classification tp, tn, fp, fn
        return loss

    def generate_train_tups(self, bert_model, train_ent1s, train_ent2s, train_links, device):
        bert_model.eval()
        all_emb1s = self.get_emb_valid(self.block_loader1, bert_model, device=device)
        all_emb2s = self.get_emb_valid(self.block_loader2, bert_model, device=device)
        train_ent_idx1s = [self.all_ent1s_p.index(e) for e in train_ent1s]
        train_ent_idx2s = [self.all_ent2s_p.index(e) for e in train_ent2s]
        train_emb1s = all_emb1s[train_ent_idx1s]
        print(Announce.printMessage(), 'all_emb1s.shape', all_emb1s.shape)
        print(Announce.printMessage(), 'train_emb1s.shape', train_emb1s.shape)
        train_emb2s = all_emb2s[train_ent_idx2s]
        print(Announce.printMessage(), 'all_emb2s.shape', all_emb2s.shape)
        print(Announce.printMessage(), 'train_emb2s.shape', train_emb2s.shape)
        # 每个entity生成一个候选实体
        candidate_dic1 = self.get_candidate_dict(train_ent1s, train_emb1s, self.all_ent2s_p, all_emb2s, device=device)
        candidate_dic2 = self.get_candidate_dict(train_ent2s, train_emb2s, self.all_ent1s_p, all_emb1s, device=device)
        train_tups = []
        for pe1, pe2 in train_links:
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    # e1
                    # 从50个相似度最高的候选队中挑出一个
                    ne1s = candidate_dic2[pe2]
                    ne1 = ne1s[np.random.randint(self.nearest_sample_num)]
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2s = candidate_dic1[pe1]
                    ne2 = ne2s[np.random.randint(self.nearest_sample_num)]
                # same check
                if pe1 != ne1 or pe2 != ne2:
                    # 添加一组positive pair和negative pair
                    train_tups.append([pe1, pe2, ne1, ne2])
            pass
        return train_tups

    def get_candidate_dict(self, train_ents, train_embs, all_ents, all_embs, device):
        topk_scores, topk_idx = self.get_topk_idx(self.nearest_sample_num, train_ents, train_embs, all_ents, all_embs,
                                                  device)
        topk_idx = topk_idx.tolist()
        candidate_dic = {train_ent: [all_ents[all_ent_idx] for all_ent_idx in all_ent_idxs] for train_ent, all_ent_idxs
                         in zip(train_ents, topk_idx)}
        return candidate_dic

    def get_topk_idx(self, topn, train_ents, train_embs, all_ents, all_embs, device):
        cos_sim_mat = cos_sim_mat_generate(train_embs, all_embs, device=device)
        print(Announce.printMessage(), 'cos_sim_mat.shape', cos_sim_mat.shape)
        # print(cos_sim_mat)
        topk_scores, topk_idx = batch_topk(cos_sim_mat, topn=topn, largest=True)
        print(Announce.printMessage(), 'topk_idx.shape:', topk_idx.shape)
        return topk_scores, topk_idx

    @staticmethod
    def get_emb_valid(loader: DataLoader, model: BertModel or t.nn.DataParallel, device='cpu') -> t.Tensor:
        results = []
        with t.no_grad():
            model.eval()
            for i, batch in TrainingTools.batch_iter(loader, 'get embedding'):
                # batch = tuple(tup.to(device) for tup in batch)
                emb = model.module.get_cls_output(batch[0].to(device), batch[1].to(device)).cpu()
                # batch = tuple(tup.cpu() for tup in batch)
                results.append(emb)
            embs = t.cat(results, dim=0)
        embs.requires_grad = False
        return embs

    @staticmethod
    def load_links(link_path, entity_ids1: dict, entity_ids2: dict):
        def links_line(line: str):
            sbj, obj = Parser.oea_truth_line(line)
            sbj = entity_ids1.get(sbj)
            obj = entity_ids2.get(obj)
            return sbj, obj

        with open(link_path, 'r', encoding='utf-8') as rfile:
            links = MPTool.packed_solver(links_line).send_packs(rfile).receive_results()
            links = list(filter(lambda x: x[0] is not None and x[1] is not None, links))
        return links

    @staticmethod
    def tids_solver(
            item,  # eid, tids
            cls_token, sep_token,
            pad_token=0,
            freqs=None,
    ):
        eid, tids = item
        assert eid is not None
        assert len(tids) > 0
        if freqs is None:
            tids = PairwisePromptTrainer.reduce_tokens(tids, max_len=seq_max_len)  # 减少token
        else:
            tids = PairwisePromptTrainer.reduce_tokens_with_freq(tids, freqs, max_len=seq_max_len)

        pad_length = seq_max_len - len(tids)
        input_ids = [cls_token] + tids + [pad_token] * pad_length
        masks = [1] * (len(tids) + 1) + [pad_token] * pad_length  # padding with 0
        assert len(input_ids) == seq_max_len + 1, len(input_ids)
        assert len(masks) == seq_max_len + 1, len(input_ids)
        # input_ids = np.array(input_ids, dtype=np.long)
        # masks = np.array(masks, dtype=np.long)
        input_ids = t.tensor(input_ids, dtype=t.long)
        masks = t.tensor(masks, dtype=t.long)
        return eid, (input_ids, masks)

    @staticmethod
    def reduce_tokens_with_freq(tids, freqs: dict, max_len=200):
        total_length = len(tids)
        if total_length <= max_len:
            return tids
        # 先标注词频和原始顺序
        tids = [(i, token, freqs.get(token)) for i, token in enumerate(tids)]
        # 再按词频大到小排序
        tids = sorted(tids, key=lambda x: x[2], reverse=False)
        while True:
            total_length = len(tids)
            if total_length <= max_len:
                break
            tids.pop()
        # 剔除后最后按原始顺序排序
        tids = sorted(tids, key=lambda x: x[0], reverse=True)
        tids = [token for i, token, freq in tids]
        return tids

    @staticmethod
    def reduce_tokens(tids, max_len=200):
        while True:
            total_length = len(tids)
            if total_length <= max_len:
                break
            tids.pop()
        return tids
