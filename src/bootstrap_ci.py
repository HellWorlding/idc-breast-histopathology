"""CPU 서버 대기열 6단계: 환자 단위 부트스트랩 신뢰구간 (val).

무엇을 하나
- 데스크톱이 보낸 val 예측 <이름>_val.npy (val_y.npy와 같은 순서)와
  서버 모델의 예측(outputs/preds, 모델·피처별 val PR-AUC 최선 조합)을 모은다.
  데스크톱 예측 위치: outputs/preds_desktop/ 또는 ~/idc-server/outputs/preds_desktop/ 중 파일이 있는 쪽.
  이름에 test가 들어간 파일은 쓰지 않는다. 함께 온 val_meta.csv가 있으면 서버 val과 행 순서가 같은지 대조한다.
- val 환자 42명을 복원 추출로 1,000번 다시 뽑아 (뽑힌 환자의 패치를 통째로 사용) 매번 PR-AUC를 계산한다.
- 모델별 val PR-AUC의 95% 구간(백분위 2.5~97.5)과, ft_color 대비 PR-AUC 차이(모델 - ft_color)의 95% 구간.
  차이는 같은 재표본에서 두 모델을 함께 계산한다(쌍 비교).

왜 환자 단위인가: 같은 환자의 패치는 서로 닮아 독립이 아니다. 패치 단위로 뽑으면 구간이 실제보다 좁게 나온다.

데스크톱 예측이 아직 없으면 아무것도 하지 않고 끝난다 (들어온 뒤 다시 실행).
출력: outputs/bootstrap_ci.csv
실행: python src/bootstrap_ci.py   (test는 열지 않는다)
"""
import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

from pathlib import Path

from common import CACHE, OUT
from cpu_models import SEED, load_groups, load_y

TAG = os.environ.get("QTAG", "boot")
REF = "ft_color"


def log(*a):
    print(f"[{TAG}]", *a, flush=True)


def one(counts, inv, y, preds):
    w = counts[inv].astype(np.float64)  # 패치별 가중치 = 그 환자가 뽑힌 횟수
    return [average_precision_score(y, p, sample_weight=w) for p in preds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dir", default="", help="데스크톱 예측 폴더 (비우면 두 후보 중 파일이 있는 쪽)")
    args = ap.parse_args()
    y, groups = load_y("val"), load_groups("val")
    cands = [Path(args.dir)] if args.dir else [OUT / "preds_desktop", Path.home() / "idc-server" / "outputs" / "preds_desktop"]
    d, files = None, []
    for c in cands:
        found = sorted(c.glob("*_val.npy")) if c.exists() else []
        if found:
            d, files = c, found
            break
    if not files:
        log(f"대기: 데스크톱 예측(*_val.npy)이 아직 없음 ({', '.join(map(str, cands))}). 들어온 뒤 다시 실행")
        return
    log(f"데스크톱 예측 폴더: {d} ({len(files)}개)")
    bad = [f.name for f in d.iterdir() if "test" in f.name.lower()]
    if bad:
        log(f"주의: 이름에 test가 들어간 파일은 읽지 않음: {bad}")
    files = [f for f in files if "test" not in f.name.lower()]
    if (d / "val_meta.csv").exists():
        # 행 대조. PR-AUC와 환자 단위 재표본에 필요한 것은 '행마다 환자와 라벨이 같은가'이다.
        # 같은 환자·같은 라벨 안에서 패치 순서가 달라도(파일 나열 순서 차이) 이 계산에는 영향이 없다.
        a = pd.read_csv(d / "val_meta.csv", dtype={"patient": str})
        b = pd.read_csv(CACHE / "val_meta.csv", dtype={"patient": str})
        if not (len(a) == len(b) and (a["patient"].values == b["patient"].values).all()
                and (a["label"].values == b["label"].values).all()):
            log("중단: 데스크톱 val_meta.csv의 환자 또는 라벨이 서버 val과 행 단위로 다름. 예측을 비교할 수 없음")
            raise SystemExit(1)

        def base(col):
            return col.str.replace(chr(92), "/", regex=False).str.rsplit("/", n=1).str[-1]

        n_diff = int((base(a["path"]).values != base(b["path"]).values).sum())
        log(f"val_meta.csv 대조: {len(a)}행 모두 환자와 라벨이 서버 val과 같음. "
            f"파일 이름이 다른 행 {n_diff}개 (같은 환자·라벨 안에서 순서만 다름: PR-AUC와 환자 단위 재표본에는 영향 없음, "
            f"패치 단위로 서버 모델과 데스크톱 모델을 짝지어 비교할 수는 없음)")
    else:
        log("주의: val_meta.csv가 없어 행 순서는 대조하지 못함 (길이만 확인)")

    models = {}
    for f in files:
        p = np.load(f).astype(np.float64).ravel()
        if len(p) != len(y):
            log(f"제외 {f.name}: 길이 {len(p)} != val {len(y)}")
            continue
        models[f.name[:-len("_val.npy")]] = ("preds_desktop", p)
    for res in ("tabular_results.csv", "basic_results.csv", "tree_results.csv", "smallcnn_results.csv"):
        if not (OUT / res).exists():
            continue
        df = pd.read_csv(OUT / res)
        for _, r in df.loc[df.groupby(["model", "features"])["val_pr_auc"].idxmax()].iterrows():
            f = OUT / "preds" / (f"{r['name']}_val.npy" if r["model"] == "smallcnn" else f"tab_{r['name']}_val.npy")
            key = f.name[:-len("_val.npy")]  # 데스크톱 폴더에 같은 이름(tab_...)이 있으면 중복으로 넣지 않는다
            if f.exists() and key not in models:
                models[key] = ("server", np.load(f).astype(np.float64).ravel())
    names = list(models)
    preds = [models[k][1] for k in names]
    pats, inv = np.unique(groups, return_inverse=True)
    log(f"모델 {len(names)}개 (데스크톱 폴더 {sum(v[0] == 'preds_desktop' for v in models.values())}개), "
        f"val 환자 {len(pats)}명, 재표본 {args.n}번")

    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(pats), np.full(len(pats), 1 / len(pats)), size=args.n)
    boot = np.array(Parallel(n_jobs=os.cpu_count())(delayed(one)(c, inv, y, preds) for c in counts))
    point = np.array([average_precision_score(y, p) for p in preds])
    has_ref = REF in models
    if not has_ref:
        log(f"{REF} 예측이 없어 차이 구간은 계산하지 않음")
    rows = []
    for j, k in enumerate(names):
        r = {"name": k, "source": models[k][0], "val_pr_auc": round(point[j], 4),
             "ci95_lo": round(np.percentile(boot[:, j], 2.5), 4), "ci95_hi": round(np.percentile(boot[:, j], 97.5), 4)}
        if has_ref:
            i = names.index(REF)
            diff = boot[:, j] - boot[:, i]
            r.update({f"diff_vs_{REF}": round(point[j] - point[i], 4),
                      "diff_ci95_lo": round(np.percentile(diff, 2.5), 4),
                      "diff_ci95_hi": round(np.percentile(diff, 97.5), 4)})
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    out["n_boot"], out["n_val_patients"] = args.n, len(pats)
    out.to_csv(OUT / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    log("요약 (bootstrap_ci.csv):")
    for line in out.drop(columns=["n_boot", "n_val_patients"]).to_string(index=False).splitlines():
        log(line)


if __name__ == "__main__":
    main()
