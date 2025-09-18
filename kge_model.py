# kge_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super().__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(torch.tensor([gamma], dtype=torch.float), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.tensor([(self.gamma.item() + self.epsilon) / hidden_dim], dtype=torch.float),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * (2 if double_entity_embedding else 1)
        self.relation_dim = hidden_dim * (2 if double_relation_embedding else 1)

        # embeddings
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(self.entity_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(self.relation_embedding, a=-self.embedding_range.item(), b=self.embedding_range.item())

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.tensor([[0.5 * self.embedding_range.item()]], dtype=torch.float))

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError(f'Unsupported model: {model_name}')
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE requires double_entity_embedding=True and double_relation_embedding=False')
        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx requires double_entity_embedding=True and double_relation_embedding=True')

    # ---- forward ----
    def forward(self, sample, mode='single'):
        if mode == 'single':
            head = self.entity_embedding[sample[:, 0]].unsqueeze(1)
            relation = self.relation_embedding[sample[:, 1]].unsqueeze(1)
            tail = self.entity_embedding[sample[:, 2]].unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            bsz, neg = head_part.size(0), head_part.size(1)
            head = self.entity_embedding[head_part.reshape(-1)].view(bsz, neg, -1)
            relation = self.relation_embedding[tail_part[:, 1]].unsqueeze(1)
            tail = self.entity_embedding[tail_part[:, 2]].unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            bsz, neg = tail_part.size(0), tail_part.size(1)
            head = self.entity_embedding[head_part[:, 0]].unsqueeze(1)
            relation = self.relation_embedding[head_part[:, 1]].unsqueeze(1)
            tail = self.entity_embedding[tail_part.reshape(-1)].view(bsz, neg, -1)
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        fn = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
        }[self.model_name]
        return fn(head, relation, tail, mode)

    # ---- models ----
    def TransE(self, head, relation, tail, mode):
        score = head + relation - tail if mode != 'head-batch' else head + (relation - tail)
        return self.gamma.item() - torch.norm(score, p=1, dim=2)

    def DistMult(self, head, relation, tail, mode):
        score = (head * relation) * tail if mode != 'head-batch' else head * (relation * tail)
        return score.sum(dim=2)

    def ComplEx(self, head, relation, tail, mode):
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_r, im_r = torch.chunk(relation, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)
        if mode == 'head-batch':
            re_score = re_r * re_t + im_r * im_t
            im_score = re_r * im_t - im_r * re_t
            score = re_h * re_score + im_h * im_score
        else:
            re_score = re_h * re_r - im_h * im_r
            im_score = re_h * im_r + im_h * re_r
            score = re_score * re_t + im_score * im_t
        return score.sum(dim=2)

    def RotatE(self, head, relation, tail, mode):
        pi = np.pi
        re_h, im_h = torch.chunk(head, 2, dim=2)
        re_t, im_t = torch.chunk(tail, 2, dim=2)
        phase_r = relation / (self.embedding_range.item() / pi)
        re_r, im_r = torch.cos(phase_r), torch.sin(phase_r)
        if mode == 'head-batch':
            re_s = re_r * re_t + im_r * im_t
            im_s = re_r * im_t - im_r * re_t
            re_s, im_s = re_s - re_h, im_s - im_h
        else:
            re_s = re_h * re_r - im_h * im_r
            im_s = re_h * im_r + im_h * re_r
            re_s, im_s = re_s - re_t, im_s - im_t
        score = torch.stack([re_s, im_s], dim=0).norm(dim=0).sum(dim=2)
        return self.gamma.item() - score

    def pRotatE(self, head, relation, tail, mode):
        pi = np.pi
        ph = head / (self.embedding_range.item() / pi)
        pr = relation / (self.embedding_range.item() / pi)
        pt = tail / (self.embedding_range.item() / pi)
        score = ph + pr - pt if mode != 'head-batch' else ph + (pr - pt)
        score = torch.abs(torch.sin(score))
        return self.gamma.item() - score.sum(dim=2) * self.modulus

    # ---- one train step ----
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda(non_blocking=True)
            negative_sample = negative_sample.cuda(non_blocking=True)
            subsampling_weight = subsampling_weight.cuda(non_blocking=True)

        negative_score = model((positive_sample, negative_sample), mode)
        if args.negative_adversarial_sampling:
            neg_log = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                       * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            neg_log = F.logsigmoid(-negative_score).mean(dim=1)

        pos_score = model(positive_sample)
        pos_log = F.logsigmoid(pos_score).squeeze(1)

        if args.uni_weight:
            pos_loss = -pos_log.mean()
            neg_loss = -neg_log.mean()
        else:
            pos_loss = - (subsampling_weight * pos_log).sum() / subsampling_weight.sum()
            neg_loss = - (subsampling_weight * neg_log).sum() / subsampling_weight.sum()

        loss = 0.5 * (pos_loss + neg_loss)

        # ---- FIXED: regularization 오타 수정 ----
        if args.regularization != 0.0:
            reg_e = model.entity_embedding.norm(p=3) ** 3
            reg_r = model.relation_embedding.norm(p=3) ** 3
            regularization = args.regularization * (reg_e + reg_r)
            loss = loss + regularization
            reg_log = {'regularization': regularization.item()}
        else:
            reg_log = {}

        loss.backward()
        optimizer.step()

        # TransE 안정화: unit-norm projection (선택)
        if args.unit_norm and model.model_name == 'TransE':
            with torch.no_grad():
                e = model.entity_embedding
                model.entity_embedding.copy_(e / (e.norm(p=2, dim=1, keepdim=True) + 1e-12))

        return {
            **reg_log,
            'positive_sample_loss': pos_loss.item(),
            'negative_sample_loss': neg_loss.item(),
            'loss': loss.item()
        }

    # ---- evaluation ----
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        model.eval()
        # Countries 특수셋은 생략: 일반 KG 평가(MRR/MR/HITS)
        from torch.utils.data import DataLoader
        from dataloader import TestDataset

        head_loader = DataLoader(
            TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch'),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
            pin_memory=args.cuda,
        )
        tail_loader = DataLoader(
            TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch'),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn,
            pin_memory=args.cuda,
        )

        logs = []
        with torch.no_grad():
            for loader in (head_loader, tail_loader):
                for positive_sample, negative_sample, filter_bias, mode in loader:
                    if args.cuda:
                        positive_sample = positive_sample.cuda(non_blocking=True)
                        negative_sample = negative_sample.cuda(non_blocking=True)
                        filter_bias = filter_bias.cuda(non_blocking=True)

                    score = model((positive_sample, negative_sample), mode)  # [B, E]
                    score += filter_bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    pos_arg = positive_sample[:, 0] if mode == 'head-batch' else positive_sample[:, 2]
                    for i in range(positive_sample.size(0)):
                        rank = (argsort[i, :] == pos_arg[i]).nonzero(as_tuple=False)
                        rank = 1 + rank.item()
                        logs.append({
                            'MRR': 1.0 / rank,
                            'MR': float(rank),
                            'HITS@1': 1.0 if rank <= 1 else 0.0,
                            'HITS@3': 1.0 if rank <= 3 else 0.0,
                            'HITS@10': 1.0 if rank <= 10 else 0.0,
                        })

        metrics = {k: sum(d[k] for d in logs) / len(logs) for k in logs[0].keys()}
        return metrics
    