"""트리/선형 모델 비교 (CPU 서버용, GPU 학습과 병렬로 돌린다).

무엇을 하나
1. 피처 만들기 (train, val만. test는 읽지 않는다)
   - resnet18: ImageNet 사전학습 ResNet18(백본 고정)의 마지막 층 직전 512차원 출력.
     입력 전처리는 GPU 학습과 같다 (50→224 확대, ImageNet 정규화: common.to_input).
   - color: 색 통계 31개 = RGB 평균 3 + 표준편차 3 + 흰 픽셀 비율 1 + 채널별 8구간 히스토그램 24.
2. 모델 비교: resnet18 피처 → 로지스틱 회귀 / RandomForest / XGBoost, color 피처 → XGBoost.
   - 하이퍼파라미터는 작은 그리드. 각 조합을 train으로 학습하고 val로 평가한다.
   - 모델별 최선 조합은 val PR-AUC로 고른다 (train.py의 에폭 선택 기준과 같음).

왜 하나
- 로지스틱 회귀 + 고정 ResNet18 피처는 GPU 베이스라인(baseline_linear)과 거의 같은 모델이다 → 두 환경 결과가 맞는지 교차 확인.
- 트리 모델이 같은 피처에서 비선형 조합으로 더 뽑아낼 수 있는지 본다.
- color 피처만으로 맞히는 정도 = "색 지름길"의 크기. CNN 성능이 이것보다 얼마나 높은지가 조직 구조를 본 몫이다.

서버 정지 대비
- 피처는 8,192개 단위 조각 파일로 저장하고, 있으면 건너뛴다.
- 조합 하나가 끝날 때마다 outputs/tabular_results.csv에 바로 한 줄 저장. 다시 실행하면 끝난 조합은 건너뛴다.

실행: python src/tabular.py                      (피처 + 전체 그리드)
      python src/tabular.py --limit 3000         (빠른 점검: 작은 부분집합, 별도 파일에 저장)
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import CACHE, OUT, ROOT, load_split, metrics, seed_all, to_input

SEED = 42
CHUNK = 8192


def grid():
    """(이름, 피처, 모델 종류, 파라미터). 이름이 같으면 이미 끝난 조합으로 보고 건너뛴다."""
    g = []
    for C in (0.01, 0.1, 1.0):
        g.append((f"logreg_resnet18_C{C}", "resnet18", "logreg", {"C": C}))
    for depth in (None, 20):
        for leaf in (1, 5):
            g.append((f"rf_resnet18_d{depth}_leaf{leaf}", "resnet18", "rf",
                      {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf}))
    for feat in ("resnet18", "color"):
        for depth in (4, 6, 8):
            for lr in (0.05, 0.1):
                g.append((f"xgb_{feat}_d{depth}_lr{lr}", feat, "xgb",
                          {"n_estimators": 500, "max_depth": depth, "learning_rate": lr}))
    return g


# ---------------- 피처 ----------------
@torch.inference_mode()
def resnet_features(x, model, threads):
    out = []
    for i in range(0, len(x), 256):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + 256]))
        out.append(model(to_input(xb, 224).contiguous(memory_format=torch.channels_last)).numpy())
    return np.concatenate(out).astype(np.float32)


def color_features(x):
    """uint8 NHWC → 31개 색 통계. 0~255 기준."""
    xf = x.reshape(len(x), -1, 3).astype(np.float32)
    mean, std = xf.mean(1), xf.std(1)
    white = (xf.min(2) > 220).mean(1, keepdims=True)  # describe.py와 같은 '거의 흰 픽셀' 기준
    hist = [np.stack([np.histogram(p[:, c], bins=8, range=(0, 256))[0] for p in xf]) / xf.shape[1]
            for c in range(3)]
    return np.concatenate([mean, std, white] + hist, axis=1).astype(np.float32)


def build_features(split, feat, x, feat_dir, threads):
    final = feat_dir / f"{split}_{feat}.npy"
    if final.exists():
        return np.load(final)
    parts_dir = feat_dir / f"parts_{split}_{feat}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    model = None
    if feat == "resnet18":
        import torchvision
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()  # 마지막 층을 떼어 512차원 피처를 꺼낸다
        model = model.eval().to(memory_format=torch.channels_last)
    n_parts = (len(x) + CHUNK - 1) // CHUNK
    t0 = time.time()
    for k in range(n_parts):
        part = parts_dir / f"{k:04d}.npy"
        if part.exists():  # 서버가 중간에 멈췄다가 다시 시작해도 끝난 조각은 건너뛴다
            continue
        xb = np.asarray(x[k * CHUNK:(k + 1) * CHUNK])
        f = resnet_features(xb, model, threads) if feat == "resnet18" else color_features(xb)
        np.save(part, f)
        print(f"  {split} {feat} part {k + 1}/{n_parts} ({time.time() - t0:.0f}s)", flush=True)
    arr = np.concatenate([np.load(parts_dir / f"{k:04d}.npy") for k in range(n_parts)])
    np.save(final, arr)
    for k in range(n_parts):
        (parts_dir / f"{k:04d}.npy").unlink()
    parts_dir.rmdir()
    return arr


# ---------------- 모델 ----------------
def make_model(kind, params, threads):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=2000))
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params, n_jobs=threads, random_state=SEED)
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             n_jobs=threads, random_state=SEED, eval_metric="logloss")
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="빠른 점검: train/val을 이만큼만 (결과는 별도 파일)")
    ap.add_argument("--only", default="", help="이름에 이 문자열이 들어간 조합만 (예: xgb_color)")
    args = ap.parse_args()
    seed_all(SEED)
    torch.set_num_threads(args.threads)

    suffix = f"_limit{args.limit}" if args.limit else ""
    feat_dir = ROOT / "data" / f"features{suffix}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    results = OUT / f"tabular_results{suffix}.csv"
    preds_dir = OUT / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # --- 데이터: train, val만. test는 이 스크립트에서 절대 열지 않는다 ---
    data = {}
    for split in ("train", "val"):
        x = np.load(CACHE / f"{split}_x.npy", mmap_mode="r")
        y = np.load(CACHE / f"{split}_y.npy")
        if args.limit:
            idx = np.sort(np.random.default_rng(SEED).choice(len(y), min(args.limit, len(y)), replace=False))
            x, y = np.asarray(x[idx]), y[idx]
        data[split] = (x, y)
    print(f"train {len(data['train'][1])} / val {len(data['val'][1])}, threads {args.threads}", flush=True)

    todo = [g for g in grid() if args.only in g[0]]
    done = set(pd.read_csv(results)["name"]) if results.exists() else set()
    feats = {}
    for name, feat, kind, params in todo:
        if name in done:
            print(f"skip {name} (이미 완료)", flush=True)
            continue
        if feat not in feats:  # 필요한 피처만 그때 만든다
            print(f"features: {feat}", flush=True)
            feats[feat] = {s: build_features(s, feat, data[s][0], feat_dir, args.threads) for s in ("train", "val")}
        Xtr, Xva = feats[feat]["train"], feats[feat]["val"]
        ytr, yva = data["train"][1], data["val"][1]

        t0 = time.time()
        model = make_model(kind, params, args.threads)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        m = metrics(yva, p, 0.5)
        row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "model": kind, "features": feat,
               "params": json.dumps(params), "n_features": Xtr.shape[1], "n_train": len(ytr), "seed": SEED,
               "total_min": round((time.time() - t0) / 60, 2), **{f"val_{k}": v for k, v in m.items()}}
        # 조합 하나 끝날 때마다 바로 저장 (서버 정지 대비)
        df = pd.DataFrame([row])
        if results.exists():
            df = pd.concat([pd.read_csv(results), df], ignore_index=True)
        df.to_csv(results, index=False, encoding="utf-8-sig")
        if not args.limit:
            np.save(preds_dir / f"tab_{name}_val.npy", p.astype(np.float32))
        print(f"{name}: val pr_auc {m['pr_auc']} roc_auc {m['roc_auc']} recall {m['recall']} "
              f"precision {m['precision']} | {row['total_min']} min", flush=True)

    # --- 요약: 모델·피처별 val PR-AUC 최선 조합 ---
    if results.exists():
        df = pd.read_csv(results)
        best = df.loc[df.groupby(["model", "features"])["val_pr_auc"].idxmax()]
        best = best.sort_values("val_pr_auc", ascending=False)
        best.to_csv(OUT / f"tabular_best{suffix}.csv", index=False, encoding="utf-8-sig")
        print(best[["name", "val_pr_auc", "val_roc_auc", "val_recall", "val_precision", "total_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
