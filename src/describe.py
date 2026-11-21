"""데이터 표 구조와 기초 통계 (df.info / describe 수준).

무엇을 하나
- 패치 표(patches_split.csv)의 열, 자료형, 결측, 고유값 수를 확인한다.
- 파일 이름에 숨은 피처(슬라이드 안 좌표 x, y)를 꺼내고, 이름 속 class와 폴더 라벨이 일치하는지 검사한다.
- 환자 단위 표(환자별 패치 수, 양성 비율, 슬라이드 크기)를 요약한다.
- 픽셀 통계(클래스별 RGB 평균, 배경 비율)는 train 캐시로만 계산한다. test는 보지 않는다.

결과: outputs/table_summary.md (README에 옮겨 적은 표의 원본), outputs/table_summary.json
실행: python src\\describe.py   (cache.py 실행 후)
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def md_table(df, index=True):
    try:
        return df.to_markdown(index=index)  # tabulate 패키지 필요
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def main():
    df = pd.read_csv(ROOT / "data" / "splits" / "patches_split.csv", dtype={"patient": str})

    # --- 1. 원래 표 구조 (df.info 결과를 그대로 남긴다) ---
    buf = io.StringIO()
    df.info(buf=buf, memory_usage="deep")
    info_text = buf.getvalue()
    schema = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "결측": df.isna().sum(),
        "고유값 수": df.nunique(),
        "예시": df.iloc[0],
    })

    # --- 2. 파일 이름에서 피처 추출: <환자>_idx5_x<X>_y<Y>_class<C>.png ---
    #     x, y는 원본 슬라이드에서 패치 왼쪽 위 좌표(픽셀). 이미지 자체 외에 표에 숨은 유일한 피처다.
    parts = df["path"].str.extract(r"(\d+)_idx5_x(\d+)_y(\d+)_class(\d)\.png$")
    df["name_patient"] = parts[0]
    df["x"] = parts[1].astype(int)
    df["y"] = parts[2].astype(int)
    df["name_class"] = parts[3].astype(int)
    checks = {
        "이름 파싱 실패": int(parts.isna().any(axis=1).sum()),
        "이름 class ≠ 폴더 라벨": int((df["name_class"] != df["label"]).sum()),
        "이름 환자 ≠ 폴더 환자": int((df["name_patient"] != df["patient"]).sum()),
        "같은 환자·좌표 중복": int(df.duplicated(["patient", "x", "y"]).sum()),
        "x, y가 50 간격 격자(1, 51, 101, ...) 밖":int(((df["x"] - 1) % 50 != 0).sum() + ((df["y"] - 1) % 50 != 0).sum()),
    }
    coord_desc = df[["x", "y"]].describe().round(0)

    # --- 3. 환자 단위 표: 분할·층화의 근거가 되는 통계 ---
    pp = df.groupby("patient").agg(
        split=("split", "first"), n=("label", "size"), pos=("label", "sum"),
        x_max=("x", "max"), y_max=("y", "max"))
    pp["pos_ratio"] = pp["pos"] / pp["n"]
    pp_desc = pp[["n", "pos", "pos_ratio", "x_max", "y_max"]].describe().round(3)
    by_split = pp.groupby("split").agg(
        환자=("n", "size"), 패치=("n", "sum"), 양성=("pos", "sum"),
        환자당_패치_중앙값=("n", "median"), 환자별_양성비율_평균=("pos_ratio", "mean"))
    by_split["양성비율"] = (by_split["양성"] / by_split["패치"]).round(4)
    by_split = by_split.loc[["train", "val", "test"]].round(3)

    # --- 4. 픽셀 통계 (train만): 색이 클래스를 얼마나 가르는지 미리 본다 ---
    #     양성이 더 보라색(핵 밀도)일 것이라는 관찰을 숫자로 확인. 모델이 색에만 기대는지 의심할 근거.
    x = np.load(ROOT / "data" / "cache" / "train_x.npy", mmap_mode="r")
    y = np.load(ROOT / "data" / "cache" / "train_y.npy")
    sums = {0: np.zeros(3), 1: np.zeros(3)}
    sq = {0: np.zeros(3), 1: np.zeros(3)}
    white = {0: 0.0, 1: 0.0}
    for i in range(0, len(x), 20000):  # 1.5GB를 한 번에 float로 올리지 않도록 나눠서
        xb = np.asarray(x[i:i + 20000], dtype=np.float64)
        yb = y[i:i + 20000]
        for c in (0, 1):
            sel = xb[yb == c]
            sums[c] += sel.mean(axis=(1, 2)).sum(0)
            sq[c] += (sel ** 2).mean(axis=(1, 2)).sum(0)
            white[c] += (sel.min(axis=3) > 220).mean(axis=(1, 2)).sum()  # 거의 흰 픽셀 = 배경/지방
    rows = []
    for c in (0, 1):
        n = int((y == c).sum())
        mean = sums[c] / n
        std = np.sqrt(sq[c] / n - mean ** 2)
        rows.append({"클래스": "IDC-" if c == 0 else "IDC+", "패치": n,
                     "R 평균": mean[0], "G 평균": mean[1], "B 평균": mean[2],
                     "R 표준편차": std[0], "G 표준편차": std[1], "B 표준편차": std[2],
                     "흰 픽셀 비율": white[c] / n})
    pix = pd.DataFrame(rows).set_index("클래스").round(3)

    # --- 저장 ---
    md = [
        "# 데이터 표 요약 (src/describe.py)", "",
        "## patches_split.csv: df.info()", "```", info_text.strip(), "```", "",
        "## 열 요약", md_table(schema), "",
        "## 파일 이름 검사", md_table(pd.Series(checks, name="건수").to_frame()), "",
        "## 좌표 피처 x, y (픽셀)", md_table(coord_desc), "",
        "## 환자 단위 통계 (279명)", md_table(pp_desc), "",
        "## 세트별 요약", md_table(by_split), "",
        "## 픽셀 통계 (train만, 0~255)", md_table(pix), "",
    ]
    (OUT / "table_summary.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "table_summary.json").write_text(json.dumps({
        "checks": checks, "coords": coord_desc.to_dict(), "patient": pp_desc.to_dict(),
        "by_split": by_split.to_dict(), "pixel_train": pix.to_dict()}, indent=2, ensure_ascii=False,
        default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
