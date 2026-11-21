"""2단계: 환자 단위 학습/검증/테스트 분할 (70/15/15).

무엇을 하나
- 패치가 아니라 '환자'를 나눈다. 한 환자의 패치는 한 세트에만 들어간다.
- 환자별 양성 비율을 5구간으로 나눠 층화한다. 세 세트의 양성 비율이 비슷해지도록.
- 겹치는 환자가 0명인지 코드로 검사하고, 결과를 CSV로 고정 저장한다.

왜 하나
- 같은 환자의 패치는 염색 색감, 조직 모양이 비슷하다. 패치 단위로 섞으면 테스트에
  '이미 본 환자'가 들어가 성능이 부풀려진다(데이터 누수).
- 분할을 파일로 고정해야 이후 모든 실험이 같은 테스트 세트로 공정하게 비교된다.
- 테스트 세트는 마지막 평가 전까지 쓰지 않는다.

실행: (.venv) PS D:\\idc-project> python src\\split.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
SEED = 42


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    patches = pd.read_csv(ROOT / "outputs" / "patches_index.csv", dtype={"patient": str})
    pp = patches.groupby("patient")["label"].agg(n="size", pos="sum").reset_index()
    pp["pos_ratio"] = pp["pos"] / pp["n"]
    # 환자별 양성 비율(1%~90%)을 5분위 구간으로 나눠 층화 기준으로 쓴다
    pp["strat"] = pd.qcut(pp["pos_ratio"], q=5, labels=False, duplicates="drop")

    # 환자를 두 번 나눈다: 70 / 30 → 30을 다시 15 / 15. 나누는 단위가 패치가 아니라 환자라는 점이 핵심
    train_p, rest_p = train_test_split(pp, test_size=0.30, stratify=pp["strat"], random_state=SEED)
    val_p, test_p = train_test_split(rest_p, test_size=0.50, stratify=rest_p["strat"], random_state=SEED)

    sets = {"train": set(train_p["patient"]), "val": set(val_p["patient"]), "test": set(test_p["patient"])}
    # 누수 검사: 어떤 환자도 두 세트에 동시에 들어가면 안 된다
    assert not (sets["train"] & sets["val"]), "train/val 환자 겹침"
    assert not (sets["train"] & sets["test"]), "train/test 환자 겹침"
    assert not (sets["val"] & sets["test"]), "val/test 환자 겹침"
    assert sum(len(s) for s in sets.values()) == len(pp), "빠진 환자 있음"

    split_of = {p: name for name, s in sets.items() for p in s}
    patches["split"] = patches["patient"].map(split_of)
    patches.to_csv(SPLITS / "patches_split.csv", index=False)
    for name, s in sets.items():
        pd.Series(sorted(s), name="patient").to_csv(SPLITS / f"{name}_patients.csv", index=False)

    report = {"seed": SEED, "overlap_patients": 0}
    for name in ("train", "val", "test"):
        sub = patches[patches["split"] == name]
        report[name] = {
            "patients": len(sets[name]),
            "patches": len(sub),
            "pos": int(sub["label"].sum()),
            "pos_ratio": round(sub["label"].mean(), 4),
        }
    (SPLITS / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
