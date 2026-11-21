"""3단계 준비: 이미지를 세트별 배열 파일(.npy)로 한 번에 묶어 둔다.

왜 하나
- 작은 PNG 27만 개를 매 에폭마다 열면 Windows에서 디스크 읽기가 병목이 된다.
- 한 번 읽어서 uint8 배열로 저장하면 이후 학습은 메모리에서 바로 읽어 빠르다.

50x50이 아닌 가장자리 패치(2,302개)
- 늘리지(resize) 않고 오른쪽/아래를 흰색(255)으로 채운다.
- 늘리면 세포 모양이 찌그러진다. 흰색은 조직이 없는 슬라이드 유리 배경과 같다.

실행: python src\\cache.py   (몇 분, train 약 1.5GB)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "splits" / "patches_split.csv", dtype={"patient": str})
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        X = np.full((len(sub), 50, 50, 3), 255, dtype=np.uint8)
        padded = 0
        for i, p in enumerate(tqdm(sub["path"], desc=split)):
            arr = np.asarray(Image.open(ROOT / p).convert("RGB"))
            h, w = arr.shape[:2]
            if (h, w) != (50, 50):
                padded += 1
            # 미리 255(흰색)로 채운 배열의 왼쪽 위에 붙인다 → 모자란 가장자리는 흰색 패딩
            X[i, :min(h, 50), :min(w, 50)] = arr[:50, :50]
        np.save(CACHE / f"{split}_x.npy", X)
        np.save(CACHE / f"{split}_y.npy", sub["label"].to_numpy(np.int64))
        sub[["patient", "path", "label"]].to_csv(CACHE / f"{split}_meta.csv", index=False)
        print(f"{split}: {len(sub)} patches, padded {padded}, saved {X.nbytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
