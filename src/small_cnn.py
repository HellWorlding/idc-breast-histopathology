"""CPU 서버 대기열 3단계: 50x50 원본 입력의 작은 CNN (Cruz-Roa 2014와 비슷한 3층 CNN).

무엇을 하나
- 합성곱 3층(32, 64, 128채널) + 완전연결 256 + 출력 1개. 사전학습 없이 처음부터 학습한다.
- 입력은 50x50 그대로, 0~1로 나눈 뒤 (x-0.5)/0.5 (데이터 통계를 쓰지 않아 정규화로 인한 누수 없음).
- 증강: 뒤집기와 90도 회전(common.augment_flip_rot). 손실: BCE. 에폭 선택: val PR-AUC.
- train으로 학습, val로 평가. test는 열지 않는다.

왜 하나
- ImageNet 사전학습 ResNet18과 색 통계 모델 사이에서, '작은 CNN이 원본 패치만으로 어디까지 가나'를 본다.

시간 관리: 첫 에폭의 처음 40배치로 에폭 시간을 추정하고, 전체가 60분을 넘을 것 같으면 에폭 수를 줄인다.
서버 정지 대비: 에폭마다 체크포인트를 저장하고, 다시 실행하면 이어서 학습한다.

색 실험 (2차 대기열): --gray 는 같은 구조에 흑백 입력을 준다. 밝기(0.299R+0.587G+0.114B)를 3채널에 똑같이 넣어
구조와 파라미터 수를 컬러와 같게 둔다. 컬러와의 차이 = 색 정보의 몫. --seed 로 시드를 바꿔 반복한다.

실행: python src/small_cnn.py
      python src/small_cnn.py --name smallcnn_gray_e30_s42 --gray --epochs 30 --seed 42 --budget 85
      python src/small_cnn.py --limit 8000 --epochs 1   (빠른 점검, 결과는 smoke_ 파일에)
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import CACHE, OUT, augment_flip_rot, metrics, save_json, seed_all

SEED = 42
TAG = os.environ.get("QTAG", "cnn")


def log(*a):
    print(f"[{TAG}]", *a, flush=True)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 50 → 25
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 25 → 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),                  # 12x12 (Grad-CAM 대상)
        )
        self.pool = nn.MaxPool2d(2)                                                            # 12 → 6
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(1)


def prep(xb, gray=False):
    """uint8 NHWC → float NCHW, -1~1. gray=True면 밝기 한 장을 3채널에 똑같이 넣는다(색 정보 제거)."""
    x = xb.permute(0, 3, 1, 2).float().div(255)
    if gray:
        x = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, 3, -1, -1)
    return (x - 0.5) / 0.5


@torch.no_grad()
def predict(model, x, bs=1024, gray=False):
    model.eval()
    return np.concatenate([torch.sigmoid(model(prep(x[i:i + bs], gray))).numpy() for i in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="smallcnn_50")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검: train을 환자 단위로 약 이만큼만")
    ap.add_argument("--budget", type=float, default=60.0, help="전체 학습 시간 한도(분)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gray", action="store_true", help="흑백 입력 (밝기를 3채널에 복제)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seed_all(args.seed)

    name = args.name + ("_smoke" if args.limit else "")
    results = OUT / ("smoke_smallcnn_results.csv" if args.limit else "smallcnn_results.csv")
    if results.exists() and name in set(pd.read_csv(results)["name"]):
        log(f"skip {name} (이미 완료)")
        return
    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / f"{name}_last.pt", ckpt_dir / f"{name}.pt"
    (OUT / "preds").mkdir(parents=True, exist_ok=True)

    # --- 데이터: train, val만 ---
    xtr = np.load(CACHE / "train_x.npy", mmap_mode="r")
    ytr = np.load(CACHE / "train_y.npy")
    if args.limit:
        from cpu_models import load_groups, patient_subsample
        idx = patient_subsample(load_groups("train"), args.limit)
        xtr, ytr = xtr[idx], ytr[idx]
    xt = torch.from_numpy(np.ascontiguousarray(xtr))
    yt = torch.from_numpy(ytr).float()
    xva = torch.from_numpy(np.load(CACHE / "val_x.npy"))
    yva = np.load(CACHE / "val_y.npy")
    n, nb = len(yt), (len(yt) + args.bs - 1) // args.bs
    log(f"{name}: train {n} / val {len(yva)}, threads {args.threads}, 에폭당 {nb}배치")

    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    st = {"epoch": 0, "max_epochs": args.epochs, "hist": [], "best_pr": -1.0, "best_epoch": 0,
          "elapsed": 0.0, "note": "", "timed": False}
    if last_path.exists():  # 이어서 학습
        ck = torch.load(last_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck["st"]
        log(f"이어서 학습: {st['epoch']}에폭까지 끝남 (최대 {st['max_epochs']})")

    while st["epoch"] < st["max_epochs"]:
        ep = st["epoch"] + 1
        torch.manual_seed(args.seed * 1000 + ep)  # 에폭마다 시드 고정 → 중간에 이어서 해도 같은 순서
        model.train()
        perm = torch.randperm(n)
        t0, run_loss = time.time(), 0.0
        for bi in range(nb):
            idx = perm[bi * args.bs:(bi + 1) * args.bs]
            xb = prep(augment_flip_rot(xt[idx]), args.gray)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yt[idx])
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            if not st["timed"] and bi == 19:
                t19 = time.time()
            if not st["timed"] and bi == min(119, nb - 1):
                # 1에폭 시간 추정: 처음 20배치(예열)를 빼고 100배치의 속도 × 배치 수, val 예측 몫으로 10% 가산
                # (1차 대기열에서는 처음 40배치로 재서 3.78분을 예상했지만 실제는 약 2분이었다)
                per = (time.time() - t19) / 100 if bi == 119 else (time.time() - t0) / (bi + 1)
                est_ep = per * nb / 60 * 1.1
                st["timed"] = True
                st["est_epoch_min"] = round(est_ep, 2)
                log(f"1에폭 예상 {est_ep:.2f}분 → {st['max_epochs']}에폭 예상 {est_ep * st['max_epochs']:.1f}분")
                if est_ep * st["max_epochs"] > args.budget * 0.92:
                    new = max(1, int(args.budget * 0.92 / est_ep))
                    st["note"] = (f"에폭 수를 {st['max_epochs']}에서 {new}로 줄임: 1에폭 예상 {est_ep:.1f}분이라 "
                                  f"전체가 {args.budget:g}분을 넘음")
                    st["max_epochs"] = new
                    log(st["note"])
        p = predict(model, xva, gray=args.gray)
        m = metrics(yva, p, 0.5)
        ep_min = (time.time() - t0) / 60
        st["elapsed"] += ep_min
        st["epoch"] = ep
        st["hist"].append({"epoch": ep, "train_loss": round(run_loss / n, 4), "epoch_min": round(ep_min, 2), **m})
        if m["pr_auc"] > st["best_pr"]:
            st["best_pr"], st["best_epoch"] = m["pr_auc"], ep
            torch.save(model.state_dict(), best_path)
            if not args.limit:
                np.save(OUT / "preds" / f"{name}_val.npy", p.astype(np.float32))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "st": st}, last_path)
        save_json({"name": name, "status": "running", **st}, OUT / "logs" / f"{name}.json")
        # 에폭이 끝날 때마다 다시 확인: 지금 속도로 남은 에폭을 다 돌면 한도를 넘는가
        avg = st["elapsed"] / ep
        if st["elapsed"] + avg * (st["max_epochs"] - ep) > args.budget:
            new = max(ep, ep + int((args.budget - st["elapsed"]) / avg))
            if new < st["max_epochs"]:
                st["note"] = (f"에폭 수를 {st['max_epochs']}에서 {new}로 줄임: {ep}에폭까지 {st['elapsed']:.1f}분 "
                              f"(에폭당 {avg:.2f}분)이라 전체가 {args.budget:g}분을 넘음")
                st["max_epochs"] = new
                log(st["note"])
        log(f"epoch {ep}/{st['max_epochs']}: train_loss {run_loss / n:.4f} val pr_auc {m['pr_auc']} roc_auc {m['roc_auc']} "
            f"recall {m['recall']} precision {m['precision']} | {ep_min:.2f} min (최선 {st['best_epoch']}에폭)")

    best = next(h for h in st["hist"] if h["epoch"] == st["best_epoch"])
    m = {k: v for k, v in best.items() if k not in ("epoch", "train_loss", "epoch_min")}
    params = {"arch": "conv32-conv64-conv128-fc256", "input": "gray(3ch copy)" if args.gray else "rgb",
              "lr": args.lr, "bs": args.bs, "aug": "flip+rot90",
              "epochs": st["max_epochs"], "best_epoch": st["best_epoch"],
              "epoch_min": round(st["elapsed"] / max(1, st["epoch"]), 2), "note": st["note"]}
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "model": "smallcnn",
           "features": "raw50_gray" if args.gray else "raw50",
           "params": json.dumps(params, ensure_ascii=False), "n_features": 50 * 50 * (1 if args.gray else 3), "n_train": n, "seed": args.seed,
           "total_min": round(st["elapsed"], 2), **{f"val_{k}": v for k, v in m.items()}}
    df = pd.DataFrame([row])
    if results.exists():
        df = pd.concat([pd.read_csv(results), df], ignore_index=True)
    df.to_csv(results, index=False, encoding="utf-8-sig")
    save_json({"name": name, "status": "done", **st}, OUT / "logs" / f"{name}.json")
    log(f"{name} 완료: 최선 {st['best_epoch']}에폭 val pr_auc {m['pr_auc']} roc_auc {m['roc_auc']} recall {m['recall']} "
        f"precision {m['precision']} f2 {m['f2']} FN {m['FN']} FP {m['FP']} | 총 {st['elapsed']:.1f}분")


if __name__ == "__main__":
    main()
