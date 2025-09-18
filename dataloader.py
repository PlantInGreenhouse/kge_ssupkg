# -*- coding: utf-8 -*-
import csv
from typing import List, Tuple, Dict, Sequence
import numpy as np
import torch

Triple = Tuple[str, str, str]

# -------------------------
# I/O
# -------------------------
def read_tsv(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, 'r', encoding='utf-8') as f:
        rd = csv.reader(f, delimiter='\t')
        for row in rd:
            if not row or len(row) < 3:
                continue
            h, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            triples.append((h, r, t))
    return triples

# -------------------------
# Mapping
# -------------------------
def build_mappings(train: List[Triple], valid: List[Triple], test: List[Triple]):
    all_nodes, all_rels = set(), set()
    for (h, r, t) in (train + valid + test):
        all_nodes.add(h); all_nodes.add(t); all_rels.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(all_nodes))}
    rel2id = {r: i for i, r in enumerate(sorted(all_rels))}

    def encode(triples: List[Triple]) -> np.ndarray:
        arr = np.zeros((len(triples), 3), dtype=np.int64)
        for i, (h, r, t) in enumerate(triples):
            arr[i, 0] = ent2id[h]
            arr[i, 1] = rel2id[r]
            arr[i, 2] = ent2id[t]
        return arr

    return ent2id, rel2id, encode

# -------------------------
# Reciprocal
# -------------------------
def add_reciprocal_np(triples: np.ndarray, nR: int) -> np.ndarray:
    """triples: np.ndarray [N,3] (h,r,t) -> concat with (t, r+nR, h)"""
    h = triples[:, 0]; r = triples[:, 1]; t = triples[:, 2]
    inv = np.stack([t, r + nR, h], axis=1)
    return np.concatenate([triples, inv], axis=0)

# -------------------------
# Filtered dicts
# -------------------------
def build_all_true_dict(all_triples_id: np.ndarray):
    """Return hr2t, rt2h maps for filtered eval."""
    hr2t: Dict[Tuple[int, int], set] = {}
    rt2h: Dict[Tuple[int, int], set] = {}
    for h, r, t in all_triples_id.tolist():
        hr2t.setdefault((h, r), set()).add(t)
        rt2h.setdefault((r, t), set()).add(h)
    return hr2t, rt2h

# -------------------------
# Training iterator
# -------------------------
class TrainIterator:
    """
    Yields tuple for common KGE train_step:
        (positive_sample: LongTensor[B,3],
         negative_sample: LongTensor[B, K],
         subsampling_weight: FloatTensor[B],
         mode: str)
    """
    def __init__(
        self,
        train_triples: np.ndarray,   # [N,3], int64
        nentity: int,
        hr2t: Dict[Tuple[int,int], set],
        rt2h: Dict[Tuple[int,int], set],
        negative_sample_size: int = 128,
        subsampling_weight: bool = True,
        mode_cycle: Sequence[str] = ('head-batch', 'tail-batch'),
        batch_size: int = 1024,
        device: torch.device = torch.device('cpu'),
    ):
        assert train_triples.ndim == 2 and train_triples.shape[1] == 3
        self.triples = train_triples
        self.nentity = nentity
        self.hr2t = hr2t
        self.rt2h = rt2h
        self.negative_sample_size = int(negative_sample_size)
        self.subsampling_weight = bool(subsampling_weight)
        self.modes = list(mode_cycle)
        self.batch_size = max(1, int(batch_size))

        self.idx = np.arange(len(self.triples), dtype=np.int64)
        self.ptr = 0
        self.mode_ptr = 0
        self.shuffle()

        if subsampling_weight:
            self.hr_freq = {}
            self.rt_freq = {}
            for h, r, t in self.triples.tolist():
                self.hr_freq[(h, r)] = self.hr_freq.get((h, r), 0) + 1
                self.rt_freq[(r, t)] = self.rt_freq.get((r, t), 0) + 1
        else:
            self.hr_freq = self.rt_freq = None

    def shuffle(self):
        np.random.shuffle(self.idx)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= len(self.idx):
            self.shuffle()

        b_idx = self.idx[self.ptr: self.ptr + self.batch_size]
        self.ptr += len(b_idx)
        pos_np = self.triples[b_idx]            # [B,3], int64
        mode = self.modes[self.mode_ptr % len(self.modes)]
        self.mode_ptr += 1

        # subsampling weight ~ 1/sqrt(freq)
        if self.subsampling_weight:
            if mode == 'tail-batch':
                sw = [1.0 / (self.hr_freq.get((h, r), 1)) for h, r, _ in pos_np.tolist()]
            else:
                sw = [1.0 / (self.rt_freq.get((r, t), 1)) for _, r, t in pos_np.tolist()]
            sw = (np.asarray(sw, dtype=np.float32)) ** 0.5
        else:
            sw = np.ones((len(b_idx),), dtype=np.float32)

        # negatives: [B, K], uniform sampling over entities
        K = self.negative_sample_size
        neg_np = np.random.randint(0, self.nentity, size=(pos_np.shape[0], K), dtype=np.int64)

        pos = torch.from_numpy(pos_np).long()              # [B,3]
        neg = torch.from_numpy(neg_np).long()              # [B,K]
        swt = torch.from_numpy(sw).float()                 # [B]
        return (pos, neg, swt, mode)

def train_data_iterator(
    train_triples: np.ndarray,
    nentity: int,
    hr2t, rt2h,
    negative_sample_size: int = 128,
    subsampling_weight: bool = True,
    mode_cycle=('head-batch','tail-batch'),
    batch_size: int = 1024,
    device: torch.device = torch.device('cpu'),
) -> TrainIterator:
    return TrainIterator(
        train_triples=train_triples,
        nentity=nentity,
        hr2t=hr2t, rt2h=rt2h,
        negative_sample_size=negative_sample_size,
        subsampling_weight=subsampling_weight,
        mode_cycle=mode_cycle,
        batch_size=batch_size,
        device=device
    )

# -------------------------
# Test dataset (filtered ranking)
# -------------------------
class TestDataset(torch.utils.data.Dataset):
    """
    For each positive triple (h, r, t), we build candidates of size [E]
    and a filter_bias vector [E] where entries that correspond to other true triples
    are set to -INF (except the actual target entity).
    """
    def __init__(self, triples: np.ndarray, all_triples: np.ndarray, nentity: int, nrelation: int, mode: str):
        assert mode in ('head-batch', 'tail-batch')
        self.triples = triples.astype(np.int64)
        self.nentity = int(nentity)
        self.nrelation = int(nrelation)
        self.mode = mode

        # Build filtered dicts once
        self.hr2t, self.rt2h = build_all_true_dict(all_triples)

    def __len__(self):
        return self.triples.shape[0]

    def __getitem__(self, idx: int):
        h, r, t = self.triples[idx].tolist()
        E = self.nentity
        candidates = torch.arange(E, dtype=torch.long)  # [E]
        bias = torch.zeros(E, dtype=torch.float)        # [E]

        if self.mode == 'tail-batch':
            true_tails = self.hr2t.get((h, r), set())
            if true_tails:
                idxs = torch.tensor(list(true_tails - {t}), dtype=torch.long)
                if idxs.numel() > 0:
                    bias[idxs] = float('-inf')
            pos = torch.tensor([h, r, t], dtype=torch.long)
            neg = candidates
        else:
            true_heads = self.rt2h.get((r, t), set())
            if true_heads:
                idxs = torch.tensor(list(true_heads - {h}), dtype=torch.long)
                if idxs.numel() > 0:
                    bias[idxs] = float('-inf')
            pos = torch.tensor([h, r, t], dtype=torch.long)
            neg = candidates

        return pos, neg, bias, self.mode

    @staticmethod
    def collate_fn(batch):
        pos = torch.stack([b[0] for b in batch], dim=0)      # [B,3]
        neg = torch.stack([b[1] for b in batch], dim=0)      # [B,E]
        bias = torch.stack([b[2] for b in batch], dim=0)     # [B,E]
        mode = batch[0][3]
        return pos, neg, bias, mode