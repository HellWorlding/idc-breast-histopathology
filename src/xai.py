"""CPU 서버 대기열 5단계: 설명 가능성(XAI). 모두 train으로 학습한 모델을 val에서 본다 (test는 열지 않는다).

1. shap    : XGB color(d4, lr0.05)에 SHAP(TreeExplainer). 어떤 색 통계가 예측을 좌우하는지.
             그림 outputs/figures/xai_shap_color_summary.png, 표 outputs/xai_shap_color_top10.csv
2. perm    : 순열 중요도(피처 하나를 섞었을 때 val PR-AUC가 얼마나 떨어지나). XGB color와 상위 트리 모델.
             그림 outputs/figures/xai_perm_<모델>.png, 표 outputs/xai_perm_<모델>.csv
3. tree    : color 피처만 쓴 깊이 3 결정 트리를 사람이 읽을 수 있는 규칙으로.
             글 outputs/xai_tree_color_d3.txt, 그림 outputs/figures/xai_tree_color_d3.png
4. gradcam : 작은 CNN에 Grad-CAM. val의 TP, FN, FP, TN 각 8장.
             그림 outputs/figures/xai_gradcam_smallcnn.png

이미 만든 결과는 건너뛴다 (다시 만들려면 파일을 지우고 실행).
실행: python src/xai.py            (--smoke: 작은 표본으로 빠른 점검, 파일 이름 앞에 smoke_)
"""
import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import CACHE, OUT, metrics
from cpu_models import (SEED, TREE_KINDS, feature_names, load_feats, load_groups, load_y, make_model,
                        patient_subsample, scores)

TAG = os.environ.get("QTAG", "xai")
FIG = OUT / "figures"
XGB_COLOR = ("xgb_color_d4_lr0.05", "color", "xgb", {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05}, 1.0)


def log(*a):
    print(f"[{TAG}]", *a, flush=True)


def fit(spec, smoke):
    name, feat, kind, params, frac = spec
    X, y, g = load_feats(feat, "train"), load_y("train"), load_groups("train")
    n = 12000 if smoke else int(len(y) * frac)
    if n < len(y):
        idx = patient_subsample(g, n)
        X, y = X[idx], y[idx]
    t0 = time.time()
    model = make_model(kind, params).fit(X, y)
    Xva, yva = load_feats(feat, "val"), load_y("val")
    ap = average_precision_score(yva, scores(model, Xva))
    log(f"{name} 다시 학습: n_train {len(y)}, val pr_auc {ap:.4f} ({time.time() - t0:.0f}s)")
    return model, Xva, yva


def top_tree_model(pre):
    """기존 tabular 결과와 2단계 결과를 합쳐 val PR-AUC가 가장 높은 트리 계열 모델 (XGB color 자신은 제외)."""
    files = [OUT / f"{pre}tree_results.csv"] + ([] if pre else [OUT / "tabular_results.csv"])
    dfs = [pd.read_csv(f) for f in files if f.exists()]
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["model"].isin(TREE_KINDS) & (df["name"] != XGB_COLOR[0])].sort_values("val_pr_auc", ascending=False)
    if df.empty:
        return None
    r = df.iloc[0]
    return (r["name"], r["features"], r["model"], json.loads(r["params"]), min(1.0, r["n_train"] / len(load_y("train"))))


# ---------------- 1. SHAP ----------------
def part_shap(pre, smoke, cache):
    fig_path, csv_path = FIG / f"{pre}xai_shap_color_summary.png", OUT / f"{pre}xai_shap_color_top10.csv"
    if fig_path.exists() and csv_path.exists():
        return log("skip shap (이미 있음)")
    if "xgb_color" not in cache:
        cache["xgb_color"] = fit(XGB_COLOR, smoke)
    model, Xva, yva = cache["xgb_color"]
    names = feature_names("color")
    try:
        import shap
        sv = np.asarray(shap.TreeExplainer(model).shap_values(Xva))
        method = "shap.TreeExplainer"
    except Exception as e:  # shap과 xgboost 버전이 안 맞으면 xgboost 내장 TreeSHAP으로 (같은 알고리즘)
        import xgboost as xgb
        sv = model.get_booster().predict(xgb.DMatrix(Xva), pred_contribs=True)[:, :-1]
        method = f"xgboost pred_contribs (shap 실패: {type(e).__name__})"
    log(f"SHAP 계산: {method}, val {sv.shape[0]}행 x 피처 {sv.shape[1]}개 (값의 단위: 로그 오즈)")
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(-mean_abs)
    rows = []
    for rank, j in enumerate(order[:10], 1):
        corr = float(np.corrcoef(Xva[:, j], sv[:, j])[0, 1]) if Xva[:, j].std() > 0 else 0.0
        rows.append({"rank": rank, "feature": names[j], "mean_abs_shap": round(float(mean_abs[j]), 4),
                     "share_of_total": round(float(mean_abs[j] / mean_abs.sum()), 4),
                     "corr_value_vs_shap": round(corr, 3),
                     "direction": "값이 클수록 IDC 쪽" if corr > 0 else "값이 클수록 정상 쪽"})
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        import shap
        plt.figure()
        shap.summary_plot(sv, pd.DataFrame(Xva, columns=names), max_display=15, show=False)
        plt.title("SHAP summary: XGB on color features (val)")
    except Exception:
        plt.figure(figsize=(7, 5))
        plt.barh([names[j] for j in order[:15]][::-1], mean_abs[order[:15]][::-1])
        plt.xlabel("mean |SHAP| (log-odds)")
        plt.title("SHAP: XGB on color features (val)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log(f"SHAP 상위 10개 ({csv_path.name}):")
    for line in pd.DataFrame(rows).to_string(index=False).splitlines():
        log(line)


# ---------------- 2. 순열 중요도 ----------------
def part_perm(pre, smoke, cache):
    from sklearn.inspection import permutation_importance
    specs = [XGB_COLOR]
    top = top_tree_model(pre)
    if top:
        specs.append(top)
        log(f"상위 트리 모델: {top[0]} (train 비율 {top[4]:.2f})")
    for spec in specs:
        name, feat = spec[0], spec[1]
        fig_path, csv_path = FIG / f"{pre}xai_perm_{name}.png", OUT / f"{pre}xai_perm_{name}.csv"
        if fig_path.exists() and csv_path.exists():
            log(f"skip perm {name} (이미 있음)")
            continue
        key = "xgb_color" if spec is XGB_COLOR else name
        if key not in cache:
            cache[key] = fit(spec, smoke)
        model, Xva, yva = cache[key]
        names = feature_names(feat)

        def scorer(est, X, y):
            return average_precision_score(y, scores(est, X))

        # 시간 재기: 채점 1번 × 피처 수 × 반복 수. 50분을 넘을 것 같으면 val을 환자 단위로 줄인다
        n_rep = 2 if smoke else 5
        t0 = time.time()
        base = scorer(model, Xva, yva)
        one = time.time() - t0
        est = one * len(names) * n_rep / 60
        note = "val 전체"
        if est > 50:
            idx = patient_subsample(load_groups("val"), int(len(yva) * 50 / est))
            Xva, yva = Xva[idx], yva[idx]
            base = scorer(model, Xva, yva)
            note = f"val을 환자 단위로 {len(idx)}행으로 줄임 (예상 {est:.0f}분 > 50분)"
        log(f"perm {name}: 채점 1번 {one:.2f}s x 피처 {len(names)} x 반복 {n_rep} = 예상 {est:.1f}분, {note}")
        t0 = time.time()
        r = permutation_importance(model, Xva, yva, scoring=scorer, n_repeats=n_rep, random_state=SEED, n_jobs=1)
        df = pd.DataFrame({"feature": names, "pr_auc_drop_mean": r.importances_mean.round(5),
                           "pr_auc_drop_std": r.importances_std.round(5)})
        df = df.sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
        df["baseline_pr_auc"], df["n_repeats"], df["n_val"] = round(base, 4), n_rep, len(yva)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        top20 = df.head(20).iloc[::-1]
        plt.figure(figsize=(7, 6))
        plt.barh(top20["feature"], top20["pr_auc_drop_mean"], xerr=top20["pr_auc_drop_std"])
        plt.xlabel("drop in val PR-AUC when the feature is shuffled")
        plt.title(f"Permutation importance: {name} (baseline {base:.4f})")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close("all")
        log(f"perm {name} 완료 ({(time.time() - t0) / 60:.1f}분, 기준 PR-AUC {base:.4f}). 상위 10개:")
        for line in df.head(10)[["rank", "feature", "pr_auc_drop_mean", "pr_auc_drop_std"]].to_string(index=False).splitlines():
            log(line)


# ---------------- 3. 깊이 3 결정 트리 규칙 ----------------
def part_tree(pre, smoke, cache):
    from sklearn.tree import export_text, plot_tree
    txt_path, fig_path = OUT / f"{pre}xai_tree_color_d3.txt", FIG / f"{pre}xai_tree_color_d3.png"
    if txt_path.exists() and fig_path.exists():
        return log("skip tree (이미 있음)")
    model, Xva, yva = fit(("dt_color_d3", "color", "dt", {"max_depth": 3}, 1.0), smoke)
    m = metrics(yva, scores(model, Xva), 0.5)
    names = feature_names("color")
    rules = export_text(model, feature_names=names, show_weights=True, decimals=3)
    head = (f"깊이 3 결정 트리, color 피처 31개만 사용 (train으로 학습)\n"
            f"val: pr_auc {m['pr_auc']} roc_auc {m['roc_auc']} recall {m['recall']} precision {m['precision']} "
            f"f2 {m['f2']} FN {m['FN']} FP {m['FP']}\n"
            f"피처 값: mean/std는 0~255, white_frac과 hist는 0~1 비율. weights = [정상 패치 수, IDC 패치 수] (train)\n\n")
    txt_path.write_text(head + rules, encoding="utf-8")
    plt.figure(figsize=(22, 9))
    plot_tree(model, feature_names=names, class_names=["normal", "IDC"], filled=True, proportion=True, fontsize=9)
    plt.title("Depth-3 decision tree on color features only")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close("all")
    for line in (head + rules).splitlines():
        log(line)


# ---------------- 4. Grad-CAM ----------------
def part_gradcam(pre, smoke, cache):
    import torch
    import torch.nn.functional as F
    from small_cnn import SmallCNN, predict, prep
    fig_path = FIG / f"{pre}xai_gradcam_smallcnn.png"
    if fig_path.exists():
        return log("skip gradcam (이미 있음)")
    name = "smallcnn_50_smoke" if smoke else "smallcnn_50"
    ckpt = OUT / "checkpoints" / f"{name}.pt"
    if not ckpt.exists():
        return log(f"gradcam 건너뜀: 체크포인트 없음 ({ckpt.name}). 3단계가 끝난 뒤 다시 실행")
    model = SmallCNN()
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    x = torch.from_numpy(np.load(CACHE / "val_x.npy"))
    y = np.load(CACHE / "val_y.npy")
    p = predict(model, x)
    pred = p >= 0.5
    groups = {"TP": pred & (y == 1), "FN": ~pred & (y == 1), "FP": pred & (y == 0), "TN": ~pred & (y == 0)}
    rng = np.random.default_rng(SEED)  # 각 칸에서 무작위 8장 (잘 나온 것만 고르지 않는다)
    fig, axes = plt.subplots(8, 8, figsize=(13, 13.5))
    for gi, (gname, mask) in enumerate(groups.items()):
        cand = np.flatnonzero(mask)
        pick = np.sort(rng.choice(cand, min(8, len(cand)), replace=False))
        xb = x[pick]
        a = model.features(prep(xb))            # 마지막 합성곱 출력 (128채널, 12x12)
        a.retain_grad()
        logit = model.head(model.pool(a)).squeeze(1)
        logit.sum().backward()                  # IDC 로짓에 대한 기울기
        w = a.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=(50, 50), mode="bilinear", align_corners=False).squeeze(1).numpy()
        cam = cam / (cam.max(axis=(1, 2), keepdims=True) + 1e-8)  # 패치별 최댓값으로 0~1
        for j in range(8):
            ax0, ax1 = axes[2 * gi, j], axes[2 * gi + 1, j]
            ax0.axis("off")
            ax1.axis("off")
            if j >= len(pick):
                continue
            ax0.imshow(xb[j].numpy())
            ax0.set_title(f"{gname} p={p[pick[j]]:.2f}", fontsize=9)
            ax1.imshow(xb[j].numpy())
            ax1.imshow(cam[j], cmap="jet", alpha=0.45, vmin=0, vmax=1)
    fig.suptitle("Grad-CAM (small CNN, last conv layer, evidence for IDC). Rows: image, then overlay, "
                 "for TP / FN / FP / TN (8 random val patches each)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130)
    plt.close("all")
    log(f"gradcam 저장: {fig_path.name} (val 기준 TP {groups['TP'].sum()}, FN {groups['FN'].sum()}, "
        f"FP {groups['FP'].sum()}, TN {groups['TN'].sum()} 중 각 8장 무작위)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parts", default="shap,perm,tree,gradcam")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    pre = "smoke_" if args.smoke else ""
    cache = {}
    for part in args.parts.split(","):
        log(f"--- {part} ---")
        {"shap": part_shap, "perm": part_perm, "tree": part_tree, "gradcam": part_gradcam}[part](pre, args.smoke, cache)


if __name__ == "__main__":
    main()
