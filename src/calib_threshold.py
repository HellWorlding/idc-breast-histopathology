"""CPU 서버 2차 대기열 1~3단계: 보정, 임계값, 환자 단위 재현율. 이미 저장된 val 예측만 쓴다 (학습 없음, test 금지).

대상 모델
- 데스크톱 CNN: preds_desktop 폴더의 <이름>_val.npy 중 tab_로 시작하지 않는 것 전부 (이름에 test가 든 파일은 쓰지 않는다).
- 서버 상위 트리 5개: tabular_results.csv + tree_results.csv에서 트리 계열의 val PR-AUC 상위 5개
  (같은 모델·피처의 설정만 다른 것은 하나로 친다).

--part calib     보정: ECE(확률 0~1을 10등분), Brier 점수, 신뢰도 그림.
                 outputs/calibration.csv, outputs/calibration_bins.csv, outputs/figures/calib_reliability_*.png
--part threshold 임계값:
                 (1) k=1,3,5,7,10마다 val 전체에서 비용 (FP + k*FN)/N 이 최소인 임계값. 이론값 1/(1+k), 0.5와 나란히.
                     F2가 최대인 임계값도. 각 줄은 common.metrics 형식 → outputs/threshold_analysis.csv
                 (2) 정직한 추정: val 환자 42명을 환자 단위로 반씩 나눠 한쪽에서 임계값을 고르고 다른 쪽에서 비용을 잰다.
                     시드 0~199로 200번 반복한 평균과 95% 구간(백분위 2.5~97.5) → outputs/threshold_split.csv
                     왜: (1)은 임계값을 고른 바로 그 데이터에서 비용을 재므로 낙관적이다.
--part patient   환자 단위: ft_color, arch_swint를 k=5 비용 최소 임계값(val 전체에서 고름)으로 잘랐을 때
                 IDC 패치가 있는 환자별 재현율. outputs/patient_recall.csv, outputs/patient_recall_summary.csv,
                 outputs/figures/calib_patient_recall_k5.png

실행: python src/calib_threshold.py --part calib   (--smoke: 반복 5번, 파일 이름 앞에 smoke_)
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CACHE, KS, OUT, metrics
from cpu_models import TREE_KINDS, load_groups, load_y

TAG = os.environ.get("QTAG", "calib")
FIG = OUT / "figures"
PATIENT_MODELS = ("ft_color", "arch_swint")


def log(*a):
    print(f"[{TAG}]", *a, flush=True)


def log_table(df):
    for line in df.to_string(index=False).splitlines():
        log(line)


# ---------------- 모델 모으기 ----------------
def desktop_dir():
    for c in (OUT / "preds_desktop", Path.home() / "idc-server" / "outputs" / "preds_desktop"):
        if c.exists() and any(c.glob("*_val.npy")):
            return c
    return None


def load_models():
    y = load_y("val")
    models = []
    d = desktop_dir()
    if d is None:
        log("주의: 데스크톱 예측 폴더가 없음 (CNN 없이 서버 트리만 분석)")
    else:
        bad = [f.name for f in d.iterdir() if "test" in f.name.lower()]
        if bad:
            log(f"주의: 이름에 test가 들어간 파일은 읽지 않음: {bad}")
        if (d / "val_meta.csv").exists():  # 행마다 환자와 라벨이 서버 val과 같은지 (bootstrap_ci.py와 같은 대조)
            a = pd.read_csv(d / "val_meta.csv", dtype={"patient": str})
            b = pd.read_csv(CACHE / "val_meta.csv", dtype={"patient": str})
            if not (len(a) == len(b) and (a["patient"].values == b["patient"].values).all()
                    and (a["label"].values == b["label"].values).all()):
                log("중단: 데스크톱 val_meta.csv의 환자 또는 라벨이 서버 val과 행 단위로 다름")
                raise SystemExit(1)
        for f in sorted(d.glob("*_val.npy")):
            name = f.name[:-len("_val.npy")]
            if "test" in f.name.lower() or name.startswith("tab_"):
                continue
            p = np.load(f).astype(np.float64).ravel()
            if len(p) != len(y):
                log(f"제외 {f.name}: 길이 {len(p)} != val {len(y)}")
                continue
            models.append((name, "desktop_cnn", p))
    dfs = [pd.read_csv(OUT / r) for r in ("tabular_results.csv", "tree_results.csv") if (OUT / r).exists()]
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["model"].isin(TREE_KINDS)].sort_values("val_pr_auc", ascending=False)
        for _, r in df.drop_duplicates(["model", "features"]).head(5).iterrows():
            f = OUT / "preds" / f"tab_{r['name']}_val.npy"
            if f.exists():
                models.append((r["name"], "server_tree", np.load(f).astype(np.float64).ravel()))
            else:
                log(f"제외 {r['name']}: 예측 파일 없음 ({f.name})")
    log(f"모델 {len(models)}개: CNN {sum(m[1] == 'desktop_cnn' for m in models)}개 "
        f"({', '.join(m[0] for m in models if m[1] == 'desktop_cnn')}), "
        f"트리 {sum(m[1] == 'server_tree' for m in models)}개 ({', '.join(m[0] for m in models if m[1] == 'server_tree')})")
    return y, models


# ---------------- 임계값 곡선 ----------------
NONE_THR = float(np.nextafter(1.0, 2.0))  # '전부 음성으로 판정'에 해당하는 임계값 (1보다 아주 조금 큼)


def curve(y, p):
    """가능한 모든 임계값(예측값마다 하나 + 전부 음성)에서의 TP, FP. 판정은 p >= 임계값."""
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ps[1:] != ps[:-1], True]  # 같은 값이 여러 개면 마지막 위치만
    return np.r_[NONE_THR, ps[last]], np.r_[0, tp[last]], np.r_[0, fp[last]]


def best_cost_thr(y, p, k, cv=None):
    thr, tp, fp = cv if cv is not None else curve(y, p)
    cost = (fp + k * (y.sum() - tp)) / len(y)
    return float(thr[int(np.argmin(cost))])


def cost_at(y, p, thr, k):
    pred = p >= thr
    return float(((pred & (y == 0)).sum() + k * (~pred & (y == 1)).sum()) / len(y))


# ---------------- 1. 보정 ----------------
def part_calib(pre, smoke):
    y, models = load_models()
    rows, bins_rows = [], []
    for name, src, p in models:
        b = np.minimum((p * 10).astype(int), 9)
        ece = 0.0
        for i in range(10):
            m = b == i
            if m.sum():
                conf, acc = p[m].mean(), y[m].mean()
                ece += m.sum() / len(y) * abs(conf - acc)
                bins_rows.append({"name": name, "source": src, "bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}",
                                  "n": int(m.sum()), "mean_pred": round(float(conf), 4),
                                  "frac_pos": round(float(acc), 4), "gap": round(float(conf - acc), 4)})
        m05 = metrics(y, p, 0.5)
        rows.append({"name": name, "source": src, "n_val": len(y), "ece10": round(ece, 4),
                     "brier": round(float(np.mean((p - y) ** 2)), 4), "mean_pred": round(float(p.mean()), 4),
                     "pos_rate": round(float(y.mean()), 4), "val_pr_auc": m05["pr_auc"],
                     "val_roc_auc": m05["roc_auc"], "val_loss": m05["loss"]})
    df = pd.DataFrame(rows).sort_values(["source", "ece10"])
    df.to_csv(OUT / f"{pre}calibration.csv", index=False, encoding="utf-8-sig")
    bins = pd.DataFrame(bins_rows)
    bins.to_csv(OUT / f"{pre}calibration_bins.csv", index=False, encoding="utf-8-sig")
    for src, title in (("desktop_cnn", "desktop CNNs"), ("server_tree", "server tree models (top 5)")):
        sub = bins[bins["source"] == src]
        if sub.empty:
            continue
        plt.figure(figsize=(6.5, 6.5))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for name, g in sub.groupby("name", sort=False):
            e = df.loc[df["name"] == name, "ece10"].iloc[0]
            plt.plot(g["mean_pred"], g["frac_pos"], marker="o", ms=3, lw=1.2, label=f"{name} (ECE {e:.3f})")
        plt.xlabel("mean predicted probability (10 equal-width bins)")
        plt.ylabel("observed IDC fraction")
        plt.title(f"Reliability diagram on val: {title}")
        plt.legend(fontsize=7, loc="upper left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / f"{pre}calib_reliability_{'cnn' if src == 'desktop_cnn' else 'tree'}.png", dpi=150)
        plt.close("all")
    log("보정 요약 (mean_pred가 pos_rate보다 크면 확률을 전반적으로 높게 낸다):")
    log_table(df)


# ---------------- 2. 임계값 ----------------
def part_threshold(pre, smoke):
    y, models = load_models()
    groups = load_groups("val")
    pats, inv = np.unique(groups, return_inverse=True)
    n_rep = 5 if smoke else 200
    rows, split_rows = [], []
    for name, src, p in models:
        cv = curve(y, p)
        thr, tp, fp = cv

        def add(rule, k, t):
            m = metrics(y, p, t)
            rows.append({"name": name, "source": src, "rule": rule, "k": k,
                         "cost_at_k": m[f"cost_k{k}"] if k else "", **{f"val_{a}": b for a, b in m.items()}})

        for k in KS:
            add("cost_min", k, best_cost_thr(y, p, k, cv))
            add("theory_1/(1+k)", k, 1 / (1 + k))
            add("fixed_0.5", k, 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec, rec = tp / np.maximum(tp + fp, 1), tp / y.sum()
            f2 = np.where(prec + rec > 0, 5 * prec * rec / (4 * prec + rec), 0.0)
        add("f2_max", "", float(thr[int(np.argmax(f2))]))

        # 정직한 추정: 환자 단위 반반 분할 n_rep번
        res = {k: [] for k in KS}
        for seed in range(n_rep):
            perm = np.random.default_rng(seed).permutation(len(pats))
            in_a = np.zeros(len(pats), bool)
            in_a[perm[:len(pats) // 2]] = True
            a = in_a[inv]
            ya, pa, yb, pb = y[a], p[a], y[~a], p[~a]
            cva = curve(ya, pa)
            for k in KS:
                t = best_cost_thr(ya, pa, k, cva)
                res[k].append((t, cost_at(ya, pa, t, k), cost_at(yb, pb, t, k),
                               cost_at(yb, pb, 0.5, k), cost_at(yb, pb, 1 / (1 + k), k)))
        for k in KS:
            r = np.array(res[k])

            def stat(v, label):
                return {f"{label}_mean": round(float(v.mean()), 4), f"{label}_lo": round(float(np.percentile(v, 2.5)), 4),
                        f"{label}_hi": round(float(np.percentile(v, 97.5)), 4)}

            split_rows.append({"name": name, "source": src, "k": k, "n_rep": n_rep, "n_val_patients": len(pats),
                               **stat(r[:, 0], "thr_selected"), "cost_select_half_mean": round(float(r[:, 1].mean()), 4),
                               **stat(r[:, 2], "cost_heldout_selected"), **stat(r[:, 3], "cost_heldout_thr0.5"),
                               **stat(r[:, 4], "cost_heldout_theory"),
                               **stat(r[:, 2] - r[:, 3], "diff_selected_minus_thr0.5"),
                               **stat(r[:, 2] - r[:, 4], "diff_selected_minus_theory")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{pre}threshold_analysis.csv", index=False, encoding="utf-8-sig")
    sp = pd.DataFrame(split_rows)
    sp.to_csv(OUT / f"{pre}threshold_split.csv", index=False, encoding="utf-8-sig")
    show = df[df["rule"].isin(["cost_min", "f2_max"])][["name", "rule", "k", "val_thr", "cost_at_k", "val_FN", "val_FP",
                                                         "val_recall", "val_precision", "val_f2"]]
    log(f"val 전체에서 고른 임계값 ({pre}threshold_analysis.csv, 낙관적):")
    log_table(show)
    log(f"환자 단위 반반 분할 {n_rep}번 ({pre}threshold_split.csv): 고르지 않은 쪽 절반에서 잰 비용")
    log_table(sp[["name", "k", "thr_selected_mean", "cost_heldout_selected_mean", "cost_heldout_selected_lo",
                  "cost_heldout_selected_hi", "cost_heldout_thr0.5_mean", "cost_heldout_theory_mean",
                  "diff_selected_minus_thr0.5_mean", "diff_selected_minus_thr0.5_lo", "diff_selected_minus_thr0.5_hi"]])


# ---------------- 3. 환자 단위 재현율 ----------------
def part_patient(pre, smoke):
    y, models = load_models()
    groups = load_groups("val")
    by = {m[0]: m[2] for m in models}
    rows, summ = [], []
    for name in PATIENT_MODELS:
        if name not in by:
            log(f"건너뜀: {name} 예측이 없음")
            continue
        p = by[name]
        t = best_cost_thr(y, p, 5)
        pred = p >= t
        rec = []
        for pat in np.unique(groups):
            m = groups == pat
            n_pos = int(y[m].sum())
            if n_pos == 0:  # IDC 패치가 없는 환자는 재현율을 정의할 수 없다
                continue
            r = float((pred[m] & (y[m] == 1)).sum() / n_pos)
            rec.append(r)
            rows.append({"name": name, "thr_k5_cost_min": round(t, 4), "patient": pat, "n_patches": int(m.sum()),
                         "n_idc_patches": n_pos, "tp": int((pred[m] & (y[m] == 1)).sum()), "recall": round(r, 4)})
        rec = np.array(rec)
        m5 = metrics(y, p, t)
        summ.append({"name": name, "thr_k5_cost_min": round(t, 4), "patch_recall": m5["recall"],
                     "patch_precision": m5["precision"], "cost_k5": m5["cost_k5"], "n_idc_patients": len(rec),
                     "recall_mean": round(float(rec.mean()), 4), "recall_median": round(float(np.median(rec)), 4),
                     "recall_min": round(float(rec.min()), 4), "recall_q25": round(float(np.percentile(rec, 25)), 4),
                     "recall_q75": round(float(np.percentile(rec, 75)), 4), "n_recall_zero": int((rec == 0).sum()),
                     "n_recall_below_0.5": int((rec < 0.5).sum()), "n_recall_below_0.8": int((rec < 0.8).sum()),
                     "n_val_patients": len(np.unique(groups))})
    if not rows:
        return
    pd.DataFrame(rows).to_csv(OUT / f"{pre}patient_recall.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summ)
    s.to_csv(OUT / f"{pre}patient_recall_summary.csv", index=False, encoding="utf-8-sig")
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(summ), figsize=(6 * len(summ), 4.2), squeeze=False)
    for ax, r in zip(axes[0], summ):
        v = d.loc[d["name"] == r["name"], "recall"]
        ax.hist(v, bins=np.linspace(0, 1, 21), edgecolor="white")
        ax.set_title(f"{r['name']}: per-patient recall at k=5 cost-min thr {r['thr_k5_cost_min']}\n"
                     f"{r['n_idc_patients']} IDC patients, min {r['recall_min']}, zero-recall {r['n_recall_zero']}", fontsize=9)
        ax.set_xlabel("recall over the patient's IDC patches")
        ax.set_ylabel("patients")
    plt.tight_layout()
    plt.savefig(FIG / f"{pre}calib_patient_recall_k5.png", dpi=150)
    plt.close("all")
    log("환자 단위 재현율 요약 (임계값은 val 전체에서 고른 것이라 낙관적):")
    log_table(s.T.reset_index())
    for name in d["name"].unique():
        log(f"{name}: 재현율이 낮은 환자 5명")
        log_table(d[d["name"] == name].sort_values("recall").head(5)[["patient", "n_patches", "n_idc_patches", "tp", "recall"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["calib", "threshold", "patient"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    pre = "smoke_" if args.smoke else ""
    {"calib": part_calib, "threshold": part_threshold, "patient": part_patient}[args.part](pre, args.smoke)


if __name__ == "__main__":
    main()
