"""학습 스크립트 (베이스라인과 이후 실험 공용).

베이스라인 실행 (마지막 층만 학습, 임계값 0.5):
  python -u src\\train.py --name baseline_linear --freeze --epochs 3
  (-u: 출력 버퍼링 끔 → 백그라운드 실행에서도 에폭 결과가 바로 보인다)
진행 상황: 에폭마다 outputs/logs/<name>.json을 덮어쓴다 (status: running → done)

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
    ap.add_argument("--aug", choices=["none", "flip", "flipcolor"], default="none",
                    help="flip: 뒤집기/90도 회전, flipcolor: + 밝기·대비·채도·색조 (common.COLOR_JITTER)")
    ap.add_argument("--pos-weight", type=float, default=1.0, help="양성 손실 가중치 (불균형 대응)")
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검용: 학습 패치 수 제한")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse()
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

    # --- 데이터: train과 val만 읽는다. test는 이 스크립트에서 절대 열지 않는다 ---
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    if args.limit:
        idx = np.random.default_rng(args.seed).choice(len(xtr), args.limit, replace=False)
        xtr, ytr = xtr[idx], ytr[idx]
    print(f"train {len(xtr)} / val {len(xva)} patches", flush=True)

    # --- 모델과 최적화: 학습할 파라미터만 AdamW에 넘기고, 스텝 단위 코사인 감소로 lr을 줄인다.
    #     pos_weight > 1이면 양성 오답에 손실을 더 줘서 FN을 줄이는 방향으로 학습한다 ---
    model = build_model(args.freeze).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(xtr) / args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=dev))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    # --- train 전체를 메모리(고정 메모리)에 올려 두고 배치 단위로 GPU에 보낸다. DataLoader가 필요 없다 ---
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr).float()
    if dev.type == "cuda":
        try:
            xtr_t = xtr_t.pin_memory()  # 고정 메모리면 GPU 전송이 빠르다
        except RuntimeError as e:  # Windows에서 1.5GB 고정 할당이 가끔 실패 → 일반 메모리로 계속 (결과 동일, 조금 느림)
            print("pin_memory 실패, 일반 메모리 사용:", str(e).splitlines()[0], flush=True)

    ckpt = OUT / "checkpoints" / f"{args.name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history, best, best_probs = [], None, None
    log_path = OUT / "logs" / f"{args.name}.json"
    t_start = time.time()
    # --- 에폭 루프: 학습 → val 평가 → val PR-AUC가 가장 좋을 때만 체크포인트 저장 ---
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
            # 증강과 정규화는 GPU에서 배치 단위로 (CPU 병목 방지). AMP(fp16)로 속도와 메모리 절약
            if args.aug in ("flip", "flipcolor"):
                xb = augment_flip_rot(xb)
            x_in = to_input(xb, args.img, color_aug=args.aug == "flipcolor")  # 색 증강은 autocast 밖(float32)에서
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(x_in).squeeze(1)
            loss = lossfn(logit.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)
        # val 평가. 임계값 0.5는 비교용 고정값이고, 실제 임계값은 나중에 저장된 val 확률로 k별로 고른다
        probs = predict(model, xva, args.img, dev)
        if dev.type == "cuda":
            torch.cuda.empty_cache()  # 평가용으로 잡은 메모리를 돌려줘서 다음 에폭 학습과 겹치지 않게
        m = metrics(yva, probs, 0.5)
        rec = {"epoch": ep, "train_loss": round(run_loss / seen, 4), "sec": round(time.time() - t0, 1), **m}
        history.append(rec)
        print(f"ep {ep}: train_loss {rec['train_loss']} | val loss {m['loss']} pr_auc {m['pr_auc']} "
              f"roc_auc {m['roc_auc']} recall {m['recall']} precision {m['precision']} f2 {m['f2']} "
              f"FN {m['FN']} FP {m['FP']} | {rec['sec']}s", flush=True)
        if best is None or m["pr_auc"] > best["pr_auc"]:
            best, best_probs = rec, probs
            torch.save(model.state_dict(), ckpt)
        # 에폭마다 지금까지의 기록을 덮어써서, 학습 도중에도 파일로 진행 상황을 볼 수 있게 한다
        save_json({"args": vars(args), "status": "running", "best_epoch": best["epoch"], "history": history}, log_path)

    # --- 기록: best 에폭의 val 확률(임계값 분석용), 에폭별 로그, experiments.csv 한 줄 ---
    preds_dir = OUT / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / f"{args.name}_val.npy", best_probs)
    save_json({"args": vars(args), "status": "done", "best_epoch": best["epoch"], "history": history}, log_path)
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": args.name, "freeze": args.freeze,
           "epochs": args.epochs, "best_epoch": best["epoch"], "lr": args.lr, "bs": args.bs, "img": args.img,
           "aug": args.aug, "pos_weight": args.pos_weight, "limit": args.limit, "seed": args.seed,
           "total_min": round((time.time() - t_start) / 60, 1),
           **{f"val_{k}": v for k, v in best.items() if k not in ("epoch", "sec")}}
    append_experiment(row)
    print("best epoch", best["epoch"], "-> experiments.csv, outputs/preds, checkpoint saved", flush=True)


if __name__ == "__main__":
    main()
