"""공통 함수: 시드 고정, 데이터 로드, 모델, 예측, 평가 지표, 실험 기록."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "outputs"
KS = (1, 5, 10, 20)  # FN 비용 = FP 비용의 k배 (학습 전에 미리 고정)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(name):
    x = np.load(CACHE / f"{name}_x.npy")
    y = np.load(CACHE / f"{name}_y.npy")
    return x, y


def to_input(xb, img_size):
    """uint8 NHWC (GPU) -> 정규화된 float NCHW. 정규화는 ImageNet 통계(사전학습과 같은 기준)를
    쓰므로 우리 데이터의 어느 세트에서도 통계를 계산하지 않는다 → 정규화로 인한 누수 없음."""
    x = xb.permute(0, 3, 1, 2).float().div_(255)
    if img_size != x.shape[-1]:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def augment_flip_rot(xb):
    """GPU에서 패치별 무작위 좌우/상하 뒤집기와 90도 회전. 병리 패치는 방향에 의미가 없다."""
    n = xb.shape[0]
    dev = xb.device
    m = torch.rand(n, device=dev) < 0.5
    xb[m] = xb[m].flip(2)
    m = torch.rand(n, device=dev) < 0.5
    xb[m] = xb[m].flip(1)
    k = torch.randint(0, 4, (n,), device=dev)
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            xb[m] = torch.rot90(xb[m], r, dims=(1, 2))
    return xb


def build_model(freeze):
    import torchvision
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    if freeze:
        for name, p in m.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return m


@torch.no_grad()
def predict(model, x_np, img_size, device, bs=1024):
    model.eval()
    probs = []
    for i in range(0, len(x_np), bs):
        xb = torch.from_numpy(x_np[i:i + bs]).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logit = model(to_input(xb, img_size)).squeeze(1)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(probs)


def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    out = {
        "thr": thr, "acc": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "f1": f1, "f2": f2, "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "loss": bce(y, p), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    for k in KS:  # 패치 1개당 기대 비용 (FP 1건 = 1, FN 1건 = k)
        out[f"cost_k{k}"] = (fp + k * fn) / len(y)
    return {key: (round(v, 4) if isinstance(v, float) else v) for key, v in out.items()}


def append_experiment(row):
    path = ROOT / "experiments.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # 엑셀/구글시트에서 한글 안 깨지게


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
