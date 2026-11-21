"""학습 스크립트 (베이스라인과 이후 실험 공용).

베이스라인 실행 (마지막 층만 학습, 임계값 0.5):
  python src\\train.py --name baseline_linear --freeze --epochs 3

규칙
- 검증(val)으로만 에폭 선택과 성능 확인. 테스트는 여기서 절대 읽지 않는다.
- 에폭 선택 기준은 PR-AUC(임계값과 무관한 지표). 임계값은 나중에 val 예측으로 따로 고른다.
- 결과는 experiments.csv에 한 줄씩 쌓이고, val 예측 확률은 outputs/preds에 저장된다.
"""
import argparse
import math
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from common import (OUT, ROOT, append_experiment, augment_flip_rot, build_model, load_split,
                    metrics, predict, save_json, seed_all, to_input)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--freeze", action="store_true", help="백본 고정, 마지막 층(fc)만 학습")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--img", type=int, default=224, help="입력 크기 (50x50을 이 크기로 키움)")
    ap.add_argument("--aug", choices=["none", "flip"], default="none")
    ap.add_argument("--pos-weight", type=float, default=1.0, help="양성 손실 가중치 (불균형 대응)")
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검용: 학습 패치 수 제한")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse()
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")

    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    if args.limit:
        idx = np.random.default_rng(args.seed).choice(len(xtr), args.limit, replace=False)
        xtr, ytr = xtr[idx], ytr[idx]
    print(f"train {len(xtr)} / val {len(xva)} patches")

    model = build_model(args.freeze).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(xtr) / args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=dev))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr).float()
    if dev.type == "cuda":
        xtr_t = xtr_t.pin_memory()

    ckpt = OUT / "checkpoints" / f"{args.name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_probs = [], None, None
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if args.freeze:
            model.eval()  # 고정한 백본의 BatchNorm 통계가 바뀌지 않게
        perm = torch.randperm(len(xtr_t))
        run_loss, seen = 0.0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = xtr_t[idx].to(dev, non_blocking=True)
            yb = ytr_t[idx].to(dev, non_blocking=True)
            if args.aug == "flip":
                xb = augment_flip_rot(xb)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(to_input(xb, args.img)).squeeze(1)
            loss = lossfn(logit.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)
        probs = predict(model, xva, args.img, dev)
        m = metrics(yva, probs, 0.5)
        rec = {"epoch": ep, "train_loss": round(run_loss / seen, 4), "sec": round(time.time() - t0, 1), **m}
        history.append(rec)
        print(f"ep {ep}: train_loss {rec['train_loss']} | val loss {m['loss']} pr_auc {m['pr_auc']} "
              f"roc_auc {m['roc_auc']} recall {m['recall']} precision {m['precision']} f2 {m['f2']} "
              f"FN {m['FN']} FP {m['FP']} | {rec['sec']}s")
        if best is None or m["pr_auc"] > best["pr_auc"]:
            best, best_probs = rec, probs
            torch.save(model.state_dict(), ckpt)

    preds_dir = OUT / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / f"{args.name}_val.npy", best_probs)
    save_json({"args": vars(args), "history": history}, OUT / "logs" / f"{args.name}.json")
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": args.name, "freeze": args.freeze,
           "epochs": args.epochs, "best_epoch": best["epoch"], "lr": args.lr, "bs": args.bs, "img": args.img,
           "aug": args.aug, "pos_weight": args.pos_weight, "limit": args.limit, "seed": args.seed,
           "total_min": round((time.time() - t_start) / 60, 1),
           **{f"val_{k}": v for k, v in best.items() if k not in ("epoch", "sec")}}
    append_experiment(row)
    print("best epoch", best["epoch"], "-> experiments.csv, outputs/preds, checkpoint saved")


if __name__ == "__main__":
    main()
