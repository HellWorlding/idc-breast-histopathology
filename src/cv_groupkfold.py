"""CPU 서버 대기열 4단계: 환자 단위 GroupKFold 5-fold 교차검증.

무엇을 하나
- train+val(환자 237명)을 환자 단위로 5조각으로 나눠, 4조각으로 학습하고 남은 1조각으로 평가한다.
  같은 환자의 패치가 학습과 평가에 동시에 들어가지 않는다. test는 열지 않는다.
- 모델: logreg C=0.01, RF depth20 leaf5, XGB resnet18 d8 lr0.05, XGB color d4 lr0.05,
  그리고 1~2단계(basic_results.csv, tree_results.csv)에서 val PR-AUC 상위 2개 (모델·피처가 서로 다른 것).
- 1~2단계에서 train을 줄여서 돌린 모델은 fold의 학습 쪽도 같은 비율로 환자 단위로 줄인다.

왜 하나
- 지금까지의 수치는 val 환자 42명 한 묶음에서 나온 것이다. fold별 값과 표준편차로
  '환자 묶음이 바뀌면 얼마나 흔들리는지'를 본다.

출력: outputs/cv_results.csv (모델×fold 한 줄씩, tabular_results.csv 열 + fold 열. val_ 열은 '평가에 쓴 fold'의 값)
      outputs/cv_summary.csv (모델별 PR-AUC, ROC-AUC의 평균과 표준편차(n-1로 나눔))
서버 정지 대비: fold 하나가 끝날 때마다 저장. 다시 실행하면 끝난 fold는 건너뛴다.

실행: python src/cv_groupkfold.py          (--smoke: 작은 표본으로 빠른 점검)
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import OUT, metrics
from cpu_models import SEED, append_row, load_feats, load_groups, load_y, make_model, patient_subsample, scores

TAG = os.environ.get("QTAG", "cv")
N_TRAIN_FULL = None


def log(*a):
    print(f"[{TAG}]", *a, flush=True)


FIXED = [
    ("logreg_resnet18_C0.01", "resnet18", "logreg", {"C": 0.01}, 1.0),
    ("rf_resnet18_d20_leaf5", "resnet18", "rf", {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 5}, 1.0),
    ("xgb_resnet18_d8_lr0.05", "resnet18", "xgb", {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05}, 1.0),
    ("xgb_color_d4_lr0.05", "color", "xgb", {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05}, 1.0),
]


def top2(prefix):
    """1~2단계 결과에서 val PR-AUC 상위 2개. 같은 모델·피처의 설정만 다른 것은 하나로 친다."""
    dfs = [pd.read_csv(OUT / f"{prefix}{s}_results.csv") for s in ("basic", "tree")
           if (OUT / f"{prefix}{s}_results.csv").exists()]
    if not dfs:
        return []
    df = pd.concat(dfs, ignore_index=True).sort_values("val_pr_auc", ascending=False)
    df = df.drop_duplicates(["model", "features"]).head(2)
    n_full = len(load_y("train"))
    return [(r["name"], r["features"], r["model"], json.loads(r["params"]),
             1.0 if prefix else min(1.0, r["n_train"] / n_full)) for _, r in df.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    prefix = "smoke_" if args.smoke else ""
    results, summary = OUT / f"{prefix}cv_results.csv", OUT / f"{prefix}cv_summary.csv"

    y = np.concatenate([load_y("train"), load_y("val")])
    groups = np.concatenate([load_groups("train"), load_groups("val")])
    keep = patient_subsample(groups, 15000) if args.smoke else np.arange(len(y))
    y, groups = y[keep], groups[keep]
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    log(f"train+val {len(y)}행, 환자 {len(np.unique(groups))}명, 5-fold (fold별 평가 환자 수: "
        f"{[len(np.unique(groups[te])) for _, te in folds]})")

    models = FIXED + top2(prefix)
    log("모델: " + ", ".join(f"{m[0]}(train 비율 {m[4]:.2f})" for m in models))
    done = set()
    if results.exists():
        d = pd.read_csv(results)
        done = set(zip(d["name"], d["fold"]))
    for name, feat, kind, params, frac in models:
        X = np.concatenate([load_feats(feat, "train"), load_feats(feat, "val")])[keep]
        for k, (tr, te) in enumerate(folds, 1):
            if (name, k) in done:
                log(f"skip {name} fold {k} (이미 완료)")
                continue
            if frac < 1.0:  # 1~2단계에서 줄여서 돌린 모델은 같은 비율로
                tr = tr[patient_subsample(groups[tr], int(len(tr) * frac))]
            t0 = time.time()
            model = make_model(kind, params)
            model.fit(X[tr], y[tr])
            m = metrics(y[te], scores(model, X[te]), 0.5)
            row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "model": kind, "features": feat,
                   "params": json.dumps(params), "n_features": X.shape[1], "n_train": len(tr), "seed": SEED,
                   "total_min": round((time.time() - t0) / 60, 2), **{f"val_{a}": b for a, b in m.items()}, "fold": k}
            append_row(results, row)
            log(f"{name} fold {k}/5: pr_auc {m['pr_auc']} roc_auc {m['roc_auc']} | n_train {len(tr)} "
                f"n_eval {len(te)} | {row['total_min']} min")

    d = pd.read_csv(results)
    rows = []
    for name, g in d.groupby("name", sort=False):
        g = g.sort_values("fold")
        r = {"name": name, "model": g["model"].iloc[0], "features": g["features"].iloc[0], "n_folds": len(g),
             "pr_auc_mean": round(g["val_pr_auc"].mean(), 4), "pr_auc_std": round(g["val_pr_auc"].std(ddof=1), 4),
             "roc_auc_mean": round(g["val_roc_auc"].mean(), 4), "roc_auc_std": round(g["val_roc_auc"].std(ddof=1), 4)}
        for _, f in g.iterrows():
            r[f"pr_auc_fold{int(f['fold'])}"] = f["val_pr_auc"]
        for _, f in g.iterrows():
            r[f"roc_auc_fold{int(f['fold'])}"] = f["val_roc_auc"]
        rows.append(r)
    s = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    s.to_csv(summary, index=False, encoding="utf-8-sig")
    log(f"요약 ({summary.name}):")
    for line in s[["name", "n_folds", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std"]].to_string(index=False).splitlines():
        log(line)


if __name__ == "__main__":
    main()
