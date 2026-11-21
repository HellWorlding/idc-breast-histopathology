"""CPU 서버 대기열 1·2단계: 기본 모델과 트리 계열 비교.

무엇을 하나
- tabular.py가 만들어 둔 피처(resnet18 512개, color 31개, 둘을 붙인 both 543개)로 여러 모델을 비교한다.
- train으로 학습하고 val로 평가한다. test(data/cache/test_*)는 이 스크립트에서 열지 않는다.
- 지표와 열 형식은 tabular_results.csv와 같다 (common.metrics, 임계값 0.5).

시간 관리 (조합마다)
- 본 학습 전에 train의 환자 단위 표본 2개(약 8천, 3만 2천 행)로 학습 시간을 재고 전체 시간을 추정한다.
- 추정이 60분을 넘으면 train을 환자 단위로 줄여 약 45분에 맞춘다. 줄였는지는 n_train 열과
  outputs/cpu_timing.csv(표본 크기, 측정 시간, 추정, 결정, 실제 시간)에 남는다.

코어 사용
- 여러 스레드를 쓰는 모델은 하나씩 순서대로 32스레드로 돌린다.
- 스레드를 하나만 쓰는 모델(결정 트리, AdaBoost, 선형 SVM, 나이브 베이즈)은 조합들을 동시에 돌린다.

서버 정지 대비: 조합 하나가 끝날 때마다 결과 CSV에 바로 한 줄 저장. 다시 실행하면 끝난 조합은 건너뛴다.

실행: python src/cpu_models.py --stage basic
      python src/cpu_models.py --stage tree
      python src/cpu_models.py --stage basic --smoke   (빠른 점검: 작은 표본, 별도 파일)
"""
import argparse
import functools
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from common import CACHE, OUT, ROOT, metrics

SEED = 42
THREADS = os.cpu_count() or 1
TAG = os.environ.get("QTAG", "cpu")
FEAT_DIR = ROOT / "data" / "features"
SINGLE = {"dt", "ada", "linsvm", "gnb"}  # 스레드를 하나만 쓰는 모델
TREE_KINDS = {"dt", "et", "ada", "hgb", "lgbm", "cat", "rf", "xgb"}


def log(*a):
    print(f"[{TAG}]", *a, flush=True)


# ---------------- 데이터 (train, val만) ----------------
def _check(split):
    if split not in ("train", "val"):
        raise ValueError(f"{split}: train과 val만 읽는다 (test 금지)")


@functools.lru_cache(maxsize=None)
def load_feats(feat, split):
    _check(split)
    if feat == "both":
        return np.concatenate([load_feats("resnet18", split), load_feats("color", split)], axis=1)
    return np.load(FEAT_DIR / f"{split}_{feat}.npy")


@functools.lru_cache(maxsize=None)
def load_y(split):
    _check(split)
    return np.load(CACHE / f"{split}_y.npy")


@functools.lru_cache(maxsize=None)
def load_groups(split):
    """패치별 환자 ID. 캐시 배열과 같은 순서 (cache.py가 같은 표에서 함께 저장)."""
    _check(split)
    return pd.read_csv(CACHE / f"{split}_meta.csv", dtype={"patient": str})["patient"].to_numpy()


def color_names():
    n = [f"mean_{c}" for c in "RGB"] + [f"std_{c}" for c in "RGB"] + ["white_frac"]
    for c in "RGB":
        n += [f"hist_{c}_{b * 32}-{b * 32 + 31}" for b in range(8)]
    return n


def feature_names(feat):
    if feat == "color":
        return color_names()
    if feat == "resnet18":
        return [f"r18_{i:03d}" for i in range(512)]
    return feature_names("resnet18") + color_names()


def patient_subsample(groups, n_target, seed=SEED):
    """환자를 무작위 순서로 하나씩 넣어 n_target행 이상이 되면 멈춘다 (환자를 쪼개지 않는다)."""
    if n_target >= len(groups):
        return np.arange(len(groups))
    pats, counts = np.unique(groups, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(pats))
    k = int(np.searchsorted(np.cumsum(counts[order]), n_target) + 1)
    return np.flatnonzero(np.isin(groups, pats[order[:k]]))


# ---------------- 모델 ----------------
def make_model(kind, p, threads=THREADS):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=2000))
    if kind == "linsvm":
        from sklearn.svm import LinearSVC
        return make_pipeline(StandardScaler(), LinearSVC(C=p["C"], dual="auto", max_iter=5000, random_state=SEED))
    if kind == "gnb":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if kind == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(solver=p["solver"], shrinkage=p.get("shrinkage"))
    if kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=p["k"], algorithm="brute", n_jobs=threads))
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        # early_stopping: train의 10%를 내부 검증으로 떼어 멈출 때를 정한다 (val은 쓰지 않는다)
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=tuple(p["hidden"]), batch_size=512,
                                           max_iter=p["max_iter"], early_stopping=True, n_iter_no_change=5,
                                           random_state=SEED))
    if kind == "rbfsvm_nys":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.svm import LinearSVC
        # RBF SVM 근사: Nystroem으로 RBF 커널 공간을 n_components차원으로 근사한 뒤 선형 SVM
        return make_pipeline(StandardScaler(),
                             Nystroem(kernel="rbf", n_components=p["n_components"], random_state=SEED, n_jobs=threads),
                             LinearSVC(C=p["C"], dual="auto", max_iter=5000, random_state=SEED))
    if kind == "dt":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=p["max_depth"], random_state=SEED)
    if kind == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**p, n_jobs=threads, random_state=SEED)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**p, n_jobs=threads, random_state=SEED)
    if kind == "ada":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(**p, random_state=SEED)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**p, early_stopping=False, random_state=SEED)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**p, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=threads, random_state=SEED, verbose=-1)
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**p, thread_count=threads, random_seed=SEED, verbose=0, allow_writing_files=False)
    if kind == "xgb":  # tabular.py와 같은 설정
        from xgboost import XGBClassifier
        return XGBClassifier(**p, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             n_jobs=threads, random_state=SEED, eval_metric="logloss")
    raise ValueError(kind)


def scores(model, X):
    """양성 점수(0~1). predict_proba가 없는 SVM은 결정 함수에 시그모이드를 씌운다:
    0.5 기준이 SVM의 원래 판정 경계(결정 함수 0)와 같고 순위가 그대로라 PR-AUC, ROC-AUC는 영향이 없다.
    다만 보정된 확률이 아니므로 SVM의 val_loss는 다른 모델과 비교하지 않는다."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = np.clip(model.decision_function(X), -50, 50)
    return 1.0 / (1.0 + np.exp(-d))


def grid(stage):
    """(이름, 피처, 모델 종류, 파라미터)"""
    g = []
    if stage == "basic":
        for feat in ("resnet18", "color"):
            for C in (0.01, 0.1, 1.0):
                g.append((f"logreg_{feat}_C{C}", feat, "logreg", {"C": C}))
            for C in (0.01, 0.1, 1.0):
                g.append((f"linsvm_{feat}_C{C}", feat, "linsvm", {"C": C}))
            for vs in (1e-9, 1e-3):
                g.append((f"gnb_{feat}_vs{vs:g}", feat, "gnb", {"var_smoothing": vs}))
            g.append((f"lda_{feat}_svd", feat, "lda", {"solver": "svd"}))
            g.append((f"lda_{feat}_lsqr_shrink", feat, "lda", {"solver": "lsqr", "shrinkage": "auto"}))
            for k in (5, 25, 101):
                g.append((f"knn_{feat}_k{k}", feat, "knn", {"k": k}))
            for hidden in ((256,), (256, 64)):
                g.append((f"mlp_{feat}_h{'x'.join(map(str, hidden))}", feat, "mlp",
                          {"hidden": list(hidden), "max_iter": 50}))
            for C in (0.1, 1.0):
                g.append((f"rbfsvm_nys_{feat}_n1000_C{C}", feat, "rbfsvm_nys", {"n_components": 1000, "C": C}))
    elif stage == "tree":
        try:
            import catboost  # noqa: F401
            has_cat = True
        except Exception as e:  # 설치가 안 됐으면 CatBoost만 뺀다
            has_cat = False
            log(f"catboost를 불러올 수 없어 건너뜀: {e}")
        for feat in ("resnet18", "color", "both"):
            for depth in (3, 5, 10, 20):
                g.append((f"dt_{feat}_d{depth}", feat, "dt", {"max_depth": depth}))
            for depth, leaf in ((None, 1), (20, 5)):
                g.append((f"et_{feat}_d{depth}_leaf{leaf}", feat, "et",
                          {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf}))
            for lr in (0.5, 1.0):
                g.append((f"ada_{feat}_n200_lr{lr}", feat, "ada", {"n_estimators": 200, "learning_rate": lr}))
            for lr in (0.05, 0.1):
                g.append((f"hgb_{feat}_it500_lr{lr}", feat, "hgb",
                          {"max_iter": 500, "learning_rate": lr, "max_leaf_nodes": 31}))
            for leaves in (31, 127):
                g.append((f"lgbm_{feat}_n500_lr0.05_leaves{leaves}", feat, "lgbm",
                          {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": leaves}))
            if has_cat:
                for depth in (6, 8):
                    g.append((f"cat_{feat}_it500_lr0.1_d{depth}", feat, "cat",
                              {"iterations": 500, "learning_rate": 0.1, "depth": depth}))
        # 기존 RF, XGB 조합은 다시 돌리지 않고, 합친 피처로 XGB 최선 설정 하나만 추가
        g.append(("xgb_both_d8_lr0.05", "both", "xgb", {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05}))
    else:
        raise ValueError(stage)
    return g


# ---------------- 조합 하나 ----------------
def _timed(kind, params, Xtr, ytr, Xva, threads):
    m = make_model(kind, params, threads)
    t = time.time()
    m.fit(Xtr, ytr)
    tf = time.time() - t
    t = time.time()
    scores(m, Xva)
    return tf, time.time() - t


def run_combo(name, feat, kind, params, smoke=False, budget=60.0, threads=THREADS):
    warnings.filterwarnings("ignore")
    Xtr, ytr, gtr = load_feats(feat, "train"), load_y("train"), load_groups("train")
    Xva, yva = load_feats(feat, "val"), load_y("val")
    if smoke:  # 빠른 점검: 환자 단위로 약 12,000행만
        i0 = patient_subsample(gtr, 12000)
        Xtr, ytr, gtr = Xtr[i0], ytr[i0], gtr[i0]
    N = len(ytr)

    # 1) 소량으로 시간 재기: 환자 단위 표본 2개
    vs = np.sort(np.random.default_rng(SEED).choice(len(yva), min(4000, len(yva)), replace=False))
    probes = []
    for n in ((2000, 5000) if smoke else (8000, 32000)):
        idx = patient_subsample(gtr, n)
        tf, tp = _timed(kind, params, Xtr[idx], ytr[idx], Xva[vs], threads)
        probes.append((len(idx), tf, tp))
    (n1, tf1, _), (n2, tf2, tp2) = probes
    # 학습 시간이 행 수의 b제곱에 비례한다고 보고 전체로 늘린다. b는 1~2로 제한 (1 미만이면 과소 추정 위험)
    b = float(np.clip(np.log(max(tf2, 1e-3) / max(tf1, 1e-3)) / np.log(n2 / n1), 1.0, 2.0)) if n2 > n1 else 1.0
    est_pred = tp2 * (len(yva) / len(vs)) * ((N / n2) if kind == "knn" else 1.0)  # kNN 예측은 train 크기에도 비례
    est = (tf2 * (N / n2) ** b + est_pred) / 60

    # 2) 60분을 넘을 것 같으면 train을 환자 단위로 줄인다
    if est > budget:
        idx = patient_subsample(gtr, int(N * (0.75 * budget / est) ** (1 / b)))
        decision = f"줄임: 예상 {est:.1f}분 > {budget:g}분이라 train을 환자 단위로 {len(idx)}/{N}행만 사용"
        Xfit, yfit = Xtr[idx], ytr[idx]
    else:
        decision = "전체 train 사용"
        Xfit, yfit = Xtr, ytr

    # 3) 본 실행
    t0 = time.time()
    model = make_model(kind, params, threads)
    model.fit(Xfit, yfit)
    p = scores(model, Xva)
    total = (time.time() - t0) / 60
    m = metrics(yva, p, 0.5)
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": name, "model": kind, "features": feat,
           "params": json.dumps(params), "n_features": Xtr.shape[1], "n_train": len(yfit), "seed": SEED,
           "total_min": round(total, 2), **{f"val_{k}": v for k, v in m.items()}}
    timing = {"name": name, "probe_n1": n1, "probe_fit_s1": round(tf1, 2), "probe_n2": n2,
              "probe_fit_s2": round(tf2, 2), "probe_pred_s2": round(tp2, 2), "exponent": round(b, 2),
              "est_full_min": round(est, 2), "decision": decision, "n_train_used": len(yfit),
              "n_train_full": N, "actual_min": round(total, 2)}
    return row, timing, p.astype(np.float32)


def append_row(path, row):
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["basic", "tree"])
    ap.add_argument("--smoke", action="store_true", help="빠른 점검: 작은 표본, 결과는 smoke_ 파일에")
    ap.add_argument("--only", default="", help="이름에 이 문자열이 들어간 조합만")
    ap.add_argument("--budget", type=float, default=60.0, help="조합 하나의 학습 시간 한도(분)")
    args = ap.parse_args()

    prefix = "smoke_" if args.smoke else ""
    results = OUT / f"{prefix}{args.stage}_results.csv"
    timing_path = OUT / f"{prefix}cpu_timing.csv"
    preds_dir = OUT / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    done = set(pd.read_csv(results)["name"]) if results.exists() else set()
    old = OUT / "tabular_results.csv"  # 이미 돌린 조합(resnet18 로지스틱 등)은 다시 돌리지 않는다
    already = set(pd.read_csv(old)["name"]) if old.exists() and not args.smoke else set()
    todo = []
    for g in grid(args.stage):
        if args.only not in g[0]:
            continue
        if g[0] in done:
            log(f"skip {g[0]} (이미 완료)")
        elif g[0] in already:
            log(f"skip {g[0]} (tabular_results.csv에 이미 있음)")
        else:
            todo.append(g)
    log(f"stage {args.stage}: 남은 조합 {len(todo)}개, threads {THREADS}, train {len(load_y('train'))} / val {len(load_y('val'))}")

    def save(row, timing, p):
        append_row(results, row)
        append_row(timing_path, timing)
        if not args.smoke:
            np.save(preds_dir / f"tab_{row['name']}_val.npy", p)
        log(f"{row['name']}: val pr_auc {row['val_pr_auc']} roc_auc {row['val_roc_auc']} recall {row['val_recall']} "
            f"precision {row['val_precision']} | {row['total_min']} min | n_train {row['n_train']} | {timing['decision']}")

    # 여러 스레드를 쓰는 모델: 하나씩 순서대로
    for name, feat, kind, params in [g for g in todo if g[2] not in SINGLE]:
        log(f"start {name}")
        save(*run_combo(name, feat, kind, params, args.smoke, args.budget))

    # 스레드를 하나만 쓰는 모델: 조합들을 동시에 (끝나는 대로 저장)
    par = [g for g in todo if g[2] in SINGLE]
    if par:
        from joblib import Parallel, delayed
        import cpu_models as cm  # 작업 프로세스가 모듈 이름으로 함수를 찾도록 (__main__의 함수는 넘길 수 없음)
        n_jobs = min(len(par), 16, THREADS)
        log(f"단일 스레드 모델 {len(par)}개를 {n_jobs}개씩 동시에: {', '.join(g[0] for g in par)}")
        out = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(cm.run_combo)(name, feat, kind, params, args.smoke, args.budget, 1)
            for name, feat, kind, params in par)
        for res in out:
            save(*res)

    if results.exists():
        df = pd.read_csv(results)
        best = df.loc[df.groupby(["model", "features"])["val_pr_auc"].idxmax()].sort_values("val_pr_auc", ascending=False)
        log(f"요약: 모델·피처별 val PR-AUC 최선 ({results.name}, 전체 {len(df)}행)")
        for line in best[["name", "val_pr_auc", "val_roc_auc", "val_recall", "val_precision", "n_train",
                          "total_min"]].to_string(index=False).splitlines():
            log(line)


if __name__ == "__main__":
    main()
