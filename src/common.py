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
KS = (1, 3, 5, 7, 10)  # FN 비용 = FP 비용의 k배. 임계값 선택 전에 고정 (처음 1,5,10,20 → 1,3,5,7,10으로 조정)
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


def to_input(xb, img_size, color_aug=False):
    """uint8 NHWC (GPU) -> 정규화된 float NCHW. 정규화는 ImageNet 통계(사전학습과 같은 기준)를
    쓰므로 우리 데이터의 어느 세트에서도 통계를 계산하지 않는다 → 정규화로 인한 누수 없음.
    color_aug=True(학습 때만)면 50x50 단계에서 색 증강을 먼저 적용한다 (작은 해상도라 빠름)."""
    x = xb.permute(0, 3, 1, 2).float().div_(255)
    if color_aug:
        x = augment_color(x)
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


# 색 증강: 패치마다 밝기·대비·채도·색조를 조금씩 무작위로 바꾼다 (GPU, float NCHW 0~1).
# 왜: train에서 양성이 모든 채널에서 평균 약 14% 어둡고 흰 픽셀이 적다(9.6% vs 28.4%).
#     밝기 ±15%는 이 차이와 비슷한 크기라, 모델이 '어두우면 양성' 같은 색 지름길에 덜 기대게 된다.
#     염색 차이(환자·기관마다 다른 H&E 색감)에 대한 강건성도 함께 노린다.
COLOR_JITTER = {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.04}
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def augment_color(x, j=COLOR_JITTER):
    n = x.shape[0]

    def factor(amount):  # 패치별 [1-amount, 1+amount] 균등 분포
        return 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * amount

    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    x = x * factor(j["brightness"])
    mean = gray.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * factor(j["contrast"]) + mean
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    x = gray + (x - gray) * factor(j["saturation"])
    # 색조: YIQ 공간에서 색 성분(I, Q)을 작은 각도만큼 회전 (각도 = hue * 2π)
    theta = (torch.rand(n, device=x.device) * 2 - 1) * j["hue"] * 2 * torch.pi
    m = _RGB2YIQ.to(x.device)
    cos, sin = theta.cos(), theta.sin()
    rot = torch.zeros(n, 3, 3, device=x.device)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
    t = torch.linalg.inv(m) @ rot @ m  # 패치별 RGB → RGB 변환 행렬
    x = torch.einsum("nij,njhw->nihw", t, x)
    return x.clamp_(0, 1)


# 모델:ImageNet 사전학습 ResNet18의 마지막 층을 출력 1개(IDC 로짓)로 바꾼다.
# freeze=True면 fc만 학습 = '사전학습 특징이 얼마나 쓸모 있나'를 보는 베이스라인
def build_model(freeze):
    import torchvision
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    if freeze:
        for name, p in m.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return m


@torch.no_grad()
def predict(model, x_np, img_size, device, bs=256):
    # bs 1024였을 때 224 입력에서 학습 메모리와 합쳐 8GB를 넘어, Windows 드라이버가 시스템 RAM으로
    # 넘기면서 학습이 수 배 느려졌다(ft_flip 첫 실행, NOTES 10절). 학습 배치와 같은 256으로 맞춘다.
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


# 평가 지표: 정확도는 양성 28%라 쉽게 부풀므로 보조로만 본다.
# 핵심은 기대 비용 = (FP + k*FN) / N. 암을 놓치는 것(FN)이 오탐(FP)보다 k배 나쁘다고 둔다
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
