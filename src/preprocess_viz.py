"""전처리 과정 정리 그림 (데이터는 바꾸지 않는다).

전처리 순서 (학습과 평가에 쓰는 것과 같은 코드 경로)
  1. 원본 PNG (보통 50x50, 가장자리는 50x27 같은 조각)          - explore.py에서 중복 폴더 제외
  2. 캐시: 50x50 uint8, 모자란 부분은 흰색(255) 패딩              - cache.py
  3. 50 → 224 bilinear 확대 (ImageNet 사전학습 모델 입력 크기)    - common.to_input
  4. 0~1로 나눈 뒤 ImageNet 평균/표준편차로 정규화                - common.to_input
  (학습 때만) 뒤집기/회전 증강                                     - common.augment_flip_rot

결과: outputs/figures/preprocessing.png
그림에는 train 패치만 쓴다 (test를 보지 않는다).
실행: python src\\preprocess_viz.py
"""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CACHE, OUT, ROOT, to_input

SEED = 42


def main():
    meta = pd.read_csv(CACHE / "train_meta.csv", dtype={"patient": str})
    x = np.load(CACHE / "train_x.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)

    # 보여 줄 예: 음성 1, 양성 1, 가장자리(50x50 아님) 1
    picks = [int(rng.choice(np.flatnonzero(meta["label"].to_numpy() == c))) for c in (0, 1)]
    for i in rng.permutation(len(meta)):
        with Image.open(ROOT / meta.at[i, "path"]) as im:
            if im.size != (50, 50):
                picks.append(int(i))
                break

    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3.2 * len(picks)))
    for r, i in enumerate(picks):
        raw = np.asarray(Image.open(ROOT / meta.at[i, "path"]).convert("RGB"))
        cached = np.asarray(x[i])
        xb = torch.from_numpy(cached[None].copy())
        up = torch.nn.functional.interpolate(xb.permute(0, 3, 1, 2).float() / 255, size=(224, 224),
                                             mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
        norm = to_input(xb, 224)[0].numpy()  # 모델에 실제로 들어가는 값
        lab = "IDC+" if meta.at[i, "label"] else "IDC-"

        axes[r, 0].imshow(raw); axes[r, 0].set_title(f"1. raw {raw.shape[1]}x{raw.shape[0]} ({lab})")
        axes[r, 1].imshow(cached); axes[r, 1].set_title("2. cache 50x50 (white pad)")
        axes[r, 2].imshow(np.clip(up, 0, 1)); axes[r, 2].set_title("3. resize 224x224")
        for c, col in enumerate("rgb"):
            axes[r, 3].hist(norm[c].ravel(), bins=40, alpha=0.5, color=col, label="RGB"[c])
        axes[r, 3].set_title(f"4. normalized values\nmean {norm.mean():.2f}, range {norm.min():.1f}~{norm.max():.1f}")
        axes[r, 3].legend(fontsize=8)
        for a in axes[r, :3]:  # 테두리를 남겨 흰 패딩 영역이 보이게
            a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    path = OUT / "figures" / "preprocessing.png"
    plt.savefig(path, dpi=110)
    print("saved", path, "rows:", [(int(i), meta.at[i, "path"].split("\\")[-1]) for i in picks])


if __name__ == "__main__":
    main()
