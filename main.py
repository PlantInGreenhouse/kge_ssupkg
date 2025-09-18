#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGE runner (FuseLinker-like schedule)
- steps-first training (--steps), with epoch fallback
- step-based validation (--eval_every_steps)
- self-adversarial negative sampling (switchable)
- reciprocal relations option
- filtered evaluation
- TransE unit-norm option

TransE
python main.py \
  --data ./data \
  --model TransE \
  --dim 200 \
  --gamma 12 \
  --steps 40000 \
  --lr 1e-3 \
  --batch 512 \
  --neg 128 \
  --test_batch_size 64 \
  --device cuda \
  --regularization 1e-6 \
  --negative_adversarial_sampling \
  --adversarial_temperature 1.0 \
  --unit_norm \
  --reciprocal \
  --eval_every_steps 1000 \
  --save_best transE_fuselike_best.pt
  
DistMult
python main.py \
  --data ./data \
  --model DistMult \
  --dim 200 \
  --gamma 12 \
  --steps 40000 \
  --lr 5e-4 \
  --batch 512 \
  --neg 128 \
  --test_batch_size 64 \
  --device cuda \
  --regularization 1e-6 \
  --negative_adversarial_sampling \
  --adversarial_temperature 1.0 \
  --reciprocal \
  --eval_every_steps 1000 \
  --save_best distmult_fuselike_best.pt
  
ComplEx
python main.py \
  --data ./data \
  --model ComplEx \
  --dim 200 \
  --gamma 12 \
  --steps 40000 \
  --lr 5e-4 \
  --batch 512 \
  --neg 128 \
  --test_batch_size 64 \
  --device cuda \
  --regularization 1e-6 \
  --negative_adversarial_sampling \
  --adversarial_temperature 1.0 \
  --reciprocal \
  --eval_every_steps 1000 \
  --save_best complex_fuselike_best.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from kge_model import KGEModel  # assumes train_step/test_step APIs
from dataloader import (
    read_tsv, build_mappings, build_all_true_dict, train_data_iterator,
    add_reciprocal_np
)

# -------------------------
# Helpers
# -------------------------
def auto_embedding_flags(model_name: str):
    # Some models use "double" embeddings (e.g., ComplEx, RotatE variants)
    if model_name == 'ComplEx':
        return True, True
    if model_name == 'RotatE':
        return True, False
    if model_name == 'pRotatE':
        return True, False
    return False, False

def resolve_device(choice: str = "auto") -> torch.device:
    c = (choice or "auto").lower()
    if c == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if c == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if c == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="FuseLinker-like KGE runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # data & model
    ap.add_argument('--data', default='./data', help='folder containing train.tsv/valid.tsv/test.tsv')
    ap.add_argument('--model', default='TransE', choices=['TransE','DistMult','ComplEx','RotatE','pRotatE'])
    ap.add_argument('--dim', type=int, default=200)
    ap.add_argument('--gamma', type=float, default=12.0)

    # schedule (steps-first; epochs is fallback)
    ap.add_argument('--steps', type=int, default=40000, help='TOTAL optimizer steps; if <=0, use --epochs')
    ap.add_argument('--epochs', type=int, default=50, help='used only when --steps <= 0')
    ap.add_argument('--eval_every_steps', type=int, default=1000, help='validate & log every N steps')

    # optimization
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=1024, help='positives per step')
    ap.add_argument('--neg', type=int, default=128, help='negative samples per positive (train)')
    ap.add_argument('--regularization', type=float, default=1e-6)
    ap.add_argument('--grad_norm', type=float, default=1.0, help='max-norm for gradient clipping (0=off)')

    # eval & device
    ap.add_argument('--test_batch_size', type=int, default=64)
    ap.add_argument('--cpu_num', type=int, default=8)
    ap.add_argument('--device', default='auto', choices=['auto','cuda','cpu','mps'])

    # training tricks
    ap.add_argument('--negative_adversarial_sampling', action='store_true')
    ap.add_argument('--adversarial_temperature', type=float, default=1.0)
    ap.add_argument('--uni_weight', action='store_true', help='disable subsampling weighting in loss')
    ap.add_argument('--unit_norm', action='store_true', help='TransE entity unit-norm projection each step')
    ap.add_argument('--reciprocal', action='store_true', help='use reciprocal relations r^{-1} during training (+filter)')

    # save
    ap.add_argument('--save_best', type=str, default='model_best.pt', help='save best-by-valid MRR')
    ap.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()

    # seed
    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # device
    dev = resolve_device(args.device)

    # 1) load raw TSV
    train_raw = read_tsv(os.path.join(args.data, 'train.tsv'))
    valid_raw = read_tsv(os.path.join(args.data, 'valid.tsv'))
    test_raw  = read_tsv(os.path.join(args.data, 'test.tsv'))

    # 2) mappings & encoding
    ent2id, rel2id, encode = build_mappings(train_raw, valid_raw, test_raw)
    train_id = encode(train_raw)
    valid_id = encode(valid_raw)
    test_id  = encode(test_raw)

    # Keep a copy of the original all triples for filtered eval
    all_id = np.concatenate([train_id, valid_id, test_id], axis=0)

    nE, nR = len(ent2id), len(rel2id)
    nR_eff = nR

    # 3) reciprocal (train + filter set)
    if args.reciprocal:
        train_id = add_reciprocal_np(train_id, nR)
        all_id   = add_reciprocal_np(all_id,   nR)
        nR_eff = nR * 2

    print(f"[Data] #E={nE}  #R={nR} (eff={nR_eff})  "
          f"#Train={len(train_id)}  #Valid={len(valid_id)}  #Test={len(test_id)}")

    # 4) model
    dbl_e, dbl_r = auto_embedding_flags(args.model)
    model = KGEModel(
        model_name=args.model,
        nentity=nE,
        nrelation=nR_eff,
        hidden_dim=args.dim,
        gamma=args.gamma,
        double_entity_embedding=dbl_e,
        double_relation_embedding=dbl_r,
    ).to(dev)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5) iterator & filtered maps
    hr2t, rt2h = build_all_true_dict(all_id)
    train_iter = train_data_iterator(
        train_triples=train_id,
        nentity=nE,
        hr2t=hr2t, rt2h=rt2h,
        negative_sample_size=args.neg,
        subsampling_weight=(not args.uni_weight),
        mode_cycle=('head-batch','tail-batch'),
        batch_size=args.batch,
        device=dev
    )

    # step/test args packed (consumed by KGEModel.{train_step,test_step})
    class S: pass
    step_args = S()
    step_args.cuda = (dev.type == 'cuda')
    step_args.negative_adversarial_sampling = args.negative_adversarial_sampling
    step_args.adversarial_temperature = args.adversarial_temperature
    step_args.uni_weight = args.uni_weight
    step_args.regularization = args.regularization
    step_args.nentity = nE
    step_args.nrelation = nR_eff
    step_args.test_batch_size = args.test_batch_size
    step_args.cpu_num = args.cpu_num
    step_args.test_log_steps = 200
    step_args.unit_norm = args.unit_norm

    # 6) schedule: steps-first
    steps_per_epoch = int(np.ceil(len(train_id) / max(1, args.batch)))
    if steps_per_epoch <= 0:
        steps_per_epoch = 1

    if args.steps and args.steps > 0:
        total_steps = int(args.steps)
        total_epochs = int(np.ceil(total_steps / steps_per_epoch))
    else:
        total_epochs = max(1, args.epochs)
        total_steps  = total_epochs * steps_per_epoch

    print(f"[Schedule] steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"eval_every_steps={args.eval_every_steps}  epochs≈{total_epochs}")

    # 7) training loop (step-based validation)
    global_step = 0
    pbar = tqdm(total=total_steps, desc=f"Train [{total_epochs} epochs ~ {total_steps} steps]",
                dynamic_ncols=True, mininterval=0.2)

    best_valid_mrr = -1.0
    best_metrics = None

    for ep in range(1, total_epochs + 1):
        losses = []
        for _ in range(steps_per_epoch):
            log = KGEModel.train_step(model, optimizer, train_iter, step_args)

            # gradient clipping
            if args.grad_norm and args.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

            losses.append(log['loss'])
            global_step += 1

            # progress
            avg_loss = float(sum(losses) / max(1, len(losses)))
            pbar.set_postfix_str(f"ep {ep}/{total_epochs} | step {global_step}/{total_steps} | loss {avg_loss:.4f}")
            pbar.update(1)

            # step-based validation
            need_eval = (args.eval_every_steps > 0) and (global_step % args.eval_every_steps == 0)
            if need_eval or global_step == total_steps:
                metrics_v = KGEModel.test_step(model, valid_id, all_id, step_args)
                tqdm.write(
                    f"[Valid @step {global_step}] "
                    f"MRR={metrics_v['MRR']:.6f} MR={metrics_v['MR']:.2f} "
                    f"H@1={metrics_v['HITS@1']:.6f} H@3={metrics_v['HITS@3']:.6f} H@10={metrics_v['HITS@10']:.6f}"
                )
                if metrics_v['MRR'] > best_valid_mrr:
                    best_valid_mrr = metrics_v['MRR']
                    best_metrics = metrics_v
                    if args.save_best:
                        torch.save({'model': model.state_dict(),
                                    'step': global_step,
                                    'metrics': best_metrics}, args.save_best)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    pbar.close()

    # 8) final test (last weights)
    metrics_t = KGEModel.test_step(model, test_id, all_id, step_args)
    print("\n[Test @Last]")
    print(f"  MRR={metrics_t['MRR']:.6f}  MR={metrics_t['MR']:.2f}  "
          f"H@1={metrics_t['HITS@1']:.6f}  H@3={metrics_t['HITS@3']:.6f}  H@10={metrics_t['HITS@10']:.6f}")

    if best_metrics is not None:
        print("\n[Best @Valid]")
        print(f"  MRR={best_metrics['MRR']:.6f}  MR={best_metrics['MR']:.2f}  "
              f"H@1={best_metrics['HITS@1']:.6f}  H@3={best_metrics['HITS@3']:.6f}  H@10={best_metrics['HITS@10']:.6f}")
        print(f"  (saved to: {args.save_best})")

if __name__ == '__main__':
    main()