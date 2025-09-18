#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal RGCN-based Link Prediction with DistMult/ComplEx decoders (PyTorch Geometric)

Input files:
  - train.tsv, valid.tsv, test.tsv   (format: head \t relation \t tail)
Run:
  python rgcn_lp.py --data_dir ./data --decoder distmult --dim 200 --layers 2 --bases 30 \
    --epochs 100 --batch_size 4096 --neg_ratio 128 --lr 1e-3 --device auto --save rgcn_dist.ckpt
"""

import argparse, os, random, math
from collections import defaultdict
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import RGCNConv

# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True

def read_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["h","r","t"])
    return df["h"].tolist(), df["r"].tolist(), df["t"].tolist()

def indexify(all_triples):
    ent2id, rel2id = {}, {}
    def get(d, k): 
        if k not in d: d[k] = len(d)
        return d[k]
    H, R, T = [], [], []
    for h, r, t in all_triples:
        H.append(get(ent2id, h))
        R.append(get(rel2id, r))
        T.append(get(ent2id, t))
    return torch.tensor(H, dtype=torch.long), torch.tensor(R, dtype=torch.long), torch.tensor(T, dtype=torch.long), ent2id, rel2id

def load_and_index(data_dir):
    Htr, Rtr, Ttr = read_tsv(os.path.join(data_dir, "train.tsv"))
    Hva, Rva, Tva = read_tsv(os.path.join(data_dir, "valid.tsv"))
    Hte, Rte, Tte = read_tsv(os.path.join(data_dir, "test.tsv"))
    triples_all = list(zip(Htr+Hva+Hte, Rtr+Rva+Rte, Ttr+Tva+Tte))
    H, R, T, ent2id, rel2id = indexify(triples_all)
    n_tr = len(Htr); n_va = len(Htr)+len(Hva)
    H_tr, R_tr, T_tr = H[:len(Htr)], R[:len(Htr)], T[:len(Htr)]
    H_va, R_va, T_va = H[len(Htr):n_va], R[len(Htr):n_va], T[len(Htr):n_va]
    H_te, R_te, T_te = H[n_va:], R[n_va:], T[n_va:]
    return (H_tr, R_tr, T_tr), (H_va, R_va, T_va), (H_te, R_te, T_te), ent2id, rel2id

def build_known_set(H, R, T):
    known = defaultdict(set)
    for h, r, t in zip(H.tolist(), R.tolist(), T.tolist()):
        known[(h, r)].add(t)
    return known

# -------------------------
# Model
# -------------------------
class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, dim=200, num_layers=2, num_bases=30, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        in_dim = dim
        for _ in range(num_layers):
            out_dim = dim
            self.convs.append(RGCNConv(in_dim, out_dim, num_relations, num_bases=num_bases))
            in_dim = out_dim

    def forward(self, edge_index, edge_type):
        x = self.emb.weight
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        return x

class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, dim):
        super().__init__()
        self.rel = nn.Embedding(num_relations, dim)

    def forward(self, enc, h, r, t):
        eh, et, rr = enc[h], enc[t], self.rel(r)
        return (eh * rr * et).sum(dim=-1)

class ComplExDecoder(nn.Module):
    def __init__(self, num_relations, dim):
        super().__init__()
        self.rel_re = nn.Embedding(num_relations, dim)
        self.rel_im = nn.Embedding(num_relations, dim)

    def forward(self, enc, h, r, t):
        d2 = enc.size(1)//2
        e_re, e_im = enc[:, :d2], enc[:, d2:]
        eh_re, eh_im = e_re[h], e_im[h]
        et_re, et_im = e_re[t], e_im[t]
        rr_re, rr_im = self.rel_re(r), self.rel_im(r)
        term1 = (eh_re * rr_re - eh_im * rr_im) * et_re
        term2 = (eh_re * rr_im + eh_im * rr_re) * et_im
        return (term1 + term2).sum(dim=-1)

class RGCNLP(nn.Module):
    def __init__(self, num_nodes, num_relations, dim, layers, bases, decoder="distmult", dropout=0.2):
        super().__init__()
        enc_dim = dim*2 if decoder == "complex" else dim
        self.encoder = RGCNEncoder(num_nodes, num_relations, enc_dim, layers, bases, dropout)
        self.decoder = DistMultDecoder(num_relations, enc_dim) if decoder=="distmult" else ComplExDecoder(num_relations, dim)

    @torch.no_grad()
    def encode(self, edge_index, edge_type):
        return self.encoder(edge_index, edge_type)

    def forward(self, edge_index, edge_type, h, r, t):
        enc = self.encoder(edge_index, edge_type)
        return self.decoder(enc, h, r, t)

# -------------------------
# Neg sampling & device
# -------------------------
def make_edge_index(H, T): 
    return torch.stack([H, T], dim=0)

def get_device(argdev):
    if argdev=="auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(argdev)

def sample_neg(h, r, t, num_entities, neg_ratio=128, device="cpu"):
    bsz = h.size(0); num_neg = bsz*neg_ratio
    h_neg = h.repeat_interleave(neg_ratio)
    r_neg = r.repeat_interleave(neg_ratio)
    mask = torch.rand(num_neg, device=device) < 0.5
    t_neg = torch.randint(0, num_entities, (num_neg,), device=device, dtype=torch.long)
    h_alt = torch.randint(0, num_entities, (num_neg,), device=device, dtype=torch.long)
    h_neg = torch.where(mask, h_neg, h_alt)
    return h_neg, r_neg, t_neg

# -------------------------
# Filtered eval
# -------------------------
@torch.no_grad()
def eval_filtered(model, device, enc, decoder, eval_triples, known_all, num_entities, k_list=(1,3,10)):
    H, R, T = eval_triples
    hits = {k: 0 for k in k_list}
    ranks = []

    def score_batch(h_idx, r_idx, candidates):
        if isinstance(decoder, DistMultDecoder):
            rr = decoder.rel(r_idx)                 # [B, D]
            eh = enc[h_idx]                         # [B, D]
            et = enc[candidates]                    # [C, D]
            return (eh * rr) @ et.t()               # [B, C]
        else:
            d2 = enc.size(1)//2
            e_re, e_im = enc[:, :d2], enc[:, d2:]
            eh_re, eh_im = e_re[h_idx], e_im[h_idx]
            et_re, et_im = e_re[candidates], e_im[candidates]
            rr_re = decoder.rel_re(r_idx)
            rr_im = decoder.rel_im(r_idx)
            term1 = (eh_re * rr_re - eh_im * rr_im) @ et_re.t()
            term2 = (eh_re * rr_im + eh_im * rr_re) @ et_im.t()
            return term1 + term2

    for i in range(len(H)):
        h, r, t = H[i].item(), R[i].item(), T[i].item()
        filtered = known_all.get((h, r), set())

        candidates = torch.arange(num_entities, device=device, dtype=torch.long)
        mask = torch.ones(num_entities, dtype=torch.bool, device=device)
        if filtered:
            bad = torch.tensor(list(filtered - {t}), device=device, dtype=torch.long)  # <<< long dtype
            mask[bad] = False
        candidates = candidates[mask]

        scores = score_batch(
            torch.tensor([h], device=device, dtype=torch.long),
            torch.tensor([r], device=device, dtype=torch.long),
            candidates
        ).squeeze(0)

        pos_idx = (candidates == t).nonzero(as_tuple=False).item()
        better = (scores > scores[pos_idx]).sum().item()
        rank = better + 1
        ranks.append(rank)
        for k in k_list:
            hits[k] += int(rank <= k)

    mrr = (1.0 / torch.tensor(ranks, dtype=torch.float, device=device)).mean().item()
    hits_at = {k: hits[k] / len(H) for k in k_list}
    return mrr, hits_at

# -------------------------
# Train
# -------------------------
def train(args):
    set_seed(args.seed)
    device = get_device(args.device)
    (Htr, Rtr, Ttr), (Hva, Rva, Tva), (Hte, Rte, Tte), ent2id, rel2id = load_and_index(args.data_dir)
    num_entities, num_relations = len(ent2id), len(rel2id)

    edge_index = make_edge_index(Htr, Ttr).to(device)
    edge_type  = Rtr.to(device)

    model = RGCNLP(num_nodes=num_entities, num_relations=num_relations,
                   dim=args.dim, layers=args.layers, bases=args.bases,
                   decoder=args.decoder, dropout=args.dropout).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    known_all = build_known_set(torch.cat([Htr, Hva, Hte]),
                                torch.cat([Rtr, Rva, Rte]),
                                torch.cat([Ttr, Tva, Tte]))

    idx = torch.arange(len(Htr), dtype=torch.long)
    steps_per_epoch = math.ceil(len(idx) / args.batch_size)

    best_mrr = float("-inf")
    patience = args.patience
    wait = 0

    with tqdm(total=args.epochs, desc="Training (epochs)", ncols=100) as pbar:
        for epoch in range(1, args.epochs+1):
            model.train()
            perm = idx[torch.randperm(len(idx))]
            losses = []

            for step in range(steps_per_epoch):
                sl = step * args.batch_size
                sr = min((step+1) * args.batch_size, len(idx))
                b = perm[sl:sr]

                h = Htr[b].to(device); r = Rtr[b].to(device); t = Ttr[b].to(device)

                pos_logit = model(edge_index, edge_type, h, r, t)
                h_neg, r_neg, t_neg = sample_neg(h, r, t, num_entities, args.neg_ratio, device=device)
                neg_logit = model(edge_index, edge_type, h_neg, r_neg, t_neg)

                y_pos = torch.ones_like(pos_logit)
                y_neg = torch.zeros_like(neg_logit)
                loss = F.binary_cross_entropy_with_logits(
                    torch.cat([pos_logit, neg_logit], dim=0),
                    torch.cat([y_pos, y_neg], dim=0)
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                losses.append(loss.item())

                avg_loss = sum(losses)/len(losses)
                pbar.set_postfix_str(f"epoch={epoch:03d} step={step+1}/{steps_per_epoch} loss={avg_loss:.4f}")

            # ---- validation: 매 10 에포크마다만 실행
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    enc = model.encode(edge_index, edge_type)
                mrr, hits = eval_filtered(
                    model, device, enc, model.decoder,
                    (Hva.to(device), Rva.to(device), Tva.to(device)),
                    known_all, num_entities, k_list=(1,3,10)
                )
                avg_loss = sum(losses)/len(losses)
                tqdm.write(f"[Val] epoch={epoch:03d} | loss={avg_loss:.4f} | "
                           f"MRR={mrr:.4f} | H@1={hits[1]:.4f} "
                           f"H@3={hits[3]:.4f} H@10={hits[10]:.4f}")

                if not math.isfinite(mrr):
                    mrr = -1e9
                if mrr > best_mrr + 1e-6:
                    best_mrr = mrr; wait = 0
                    if args.save: torch.save(model.state_dict(), args.save)
                else:
                    wait += 1
                    if wait >= patience:
                        tqdm.write(f"Early stopping at epoch {epoch} (best Val MRR={best_mrr:.4f})")
                        break

            pbar.update(1)

    # ---- test
    if args.save and os.path.exists(args.save):
        model.load_state_dict(torch.load(args.save, map_location=device))
    model.eval()
    with torch.no_grad():
        enc = model.encode(edge_index, edge_type)
    mrr, hits = eval_filtered(
        model, device, enc, model.decoder,
        (Hte.to(device), Rte.to(device), Tte.to(device)),
        known_all, num_entities, k_list=(1,3,10)
    )
    print(f"[TEST] MRR={mrr:.4f} | H@1={hits[1]:.4f} H@3={hits[3]:.4f} H@10={hits[10]:.4f}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--decoder", type=str, default="distmult", choices=["distmult","complex"])
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bases", type=int, default=30)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--neg_ratio", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")  # auto/cpu/cuda/mps
    ap.add_argument("--save", type=str, default="rgcn_lp.ckpt")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()