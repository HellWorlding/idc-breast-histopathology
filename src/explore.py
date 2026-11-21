"""1단계: 데이터 기본 정보 확인.

무엇을 하나
- 환자 폴더(data/raw/<환자ID>/<0|1>/*.png)를 돌며 환자별 음성/양성 패치 수를 센다.
- 이미지 크기가 50x50이 아닌 패치(슬라이드 가장자리)를 센다.
- 양성/음성 샘플 이미지를 한 장의 그림으로 저장한다.

왜 하나
- 클래스 비율과 환자별 차이를 알아야 분할 방법(환자 단위, 층화)과 평가 지표를 정할 수 있다.
- 중복 폴더(IDC_regular_ps50_idx5)는 같은 패치의 사본이라 제외한다. 섞이면 누수가 생긴다.

실행: (.venv) PS D:\\idc-project> python src\\explore.py
"""
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
EXCLUDE = {"IDC_regular_ps50_idx5"}
SEED = 42


def collect():
    rows = []
    # 중복 폴더(IDC_regular_ps50_idx5)를 빼고 환자 폴더만 센다. 폴더 이름 0/1이 라벨
    patient_dirs = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name not in EXCLUDE)
    for pdir in tqdm(patient_dirs, desc="patients"):
        for label in ("0", "1"):
            for f in (pdir / label).glob("*.png"):
                rows.append({"patient": pdir.name, "label": int(label), "path": str(f.relative_to(ROOT))})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        raise SystemExit("패치를 찾지 못함: data/raw 아래에 압축을 풀었는지 확인")

    # 이미지 크기 확인 (헤더만 읽어서 빠름)
    sizes = Counter()
    for p in tqdm(df["path"], desc="sizes"):
        with Image.open(ROOT / p) as im:
            sizes[im.size] += 1
    df.to_csv(OUT / "patches_index.csv", index=False)

    # 환자 단위 집계: 환자마다 양성 비율이 크게 달라 이후 층화 분할의 근거가 된다
    per_patient = df.groupby("patient")["label"].agg(n="size", pos="sum")
    per_patient["neg"] = per_patient["n"] - per_patient["pos"]
    per_patient["pos_ratio"] = per_patient["pos"] / per_patient["n"]
    per_patient.to_csv(OUT / "patient_stats.csv")

    n = len(df); pos = int(df["label"].sum())
    summary = {
        "patients": int(df["patient"].nunique()),
        "patches": n,
        "neg": n - pos,
        "pos": pos,
        "pos_ratio": round(pos / n, 4),
        "all_negative_baseline_accuracy": round((n - pos) / n, 4),
        "patches_per_patient": per_patient["n"].describe().round(1).to_dict(),
        "pos_ratio_per_patient": per_patient["pos_ratio"].describe().round(3).to_dict(),
        "image_sizes_top5": {f"{w}x{h}": c for (w, h), c in sizes.most_common(5)},
        "non_50x50": int(sum(c for s, c in sizes.items() if s != (50, 50))),
    }
    (OUT / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 샘플 그림: 위 2줄 음성, 아래 2줄 양성
    rng = random.Random(SEED)
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
    for row in range(4):
        label = 0 if row < 2 else 1
        picks = rng.sample(list(df.loc[df["label"] == label, "path"]), 8)
        for col, p in enumerate(picks):
            ax = axes[row, col]
            ax.imshow(Image.open(ROOT / p).convert("RGB"))
            ax.axis("off")
        axes[row, 0].set_title("IDC-" if label == 0 else "IDC+", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "samples.png", dpi=120)

    # 환자별 패치 수, 양성 비율 분포
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(per_patient["n"], bins=30); axes[0].set_title("patches per patient")
    axes[1].hist(per_patient["pos_ratio"], bins=30); axes[1].set_title("positive ratio per patient")
    plt.tight_layout()
    plt.savefig(FIG / "patient_distribution.png", dpi=120)
    print("saved:", FIG / "samples.png", FIG / "patient_distribution.png")


if __name__ == "__main__":
    main()
