# IDC 유방 병리 패치 분류: 암을 놓치는 비용을 고려한 임계값 선택

유방 병리 슬라이드의 50×50 패치를 보고 침윤성 유관암(IDC)이 있는지 분류한다.
정확도보다 **암을 놓치는 오류(FN)**를 줄이는 데 초점을 두고, 데이터 누수를 막는 분할과 판단 근거 확인까지 포함한다.

## 데이터
- Kaggle [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) (CC0-1.0). 이 저장소에는 데이터를 넣지 않았다
- 환자 279명, 패치 277,524개, 양성 28.39% (직접 셈, `src/explore.py`)
- 원본 zip 안에 같은 패치의 사본 폴더(`IDC_regular_ps50_idx5`)가 있어 제외했다. 섞이면 같은 패치가 학습과 테스트에 동시에 들어간다

## 방법
1. **환자 단위 분할** 70/15/15, 환자별 양성 비율로 층화, 세트 간 겹치는 환자 0명 검사 (`src/split.py`)
2. **평가 기준을 먼저 고정**: FN 비용 = FP 비용의 k배 (k = 1, 5, 10, 20), 주 지표는 기대 비용, 보조 지표는 재현율, 정밀도, F2, PR-AUC
3. ImageNet 사전학습 ResNet18 전이학습 (`src/train.py`)
4. 임계값과 하이퍼파라미터는 검증 세트로만 정하고, 테스트 세트는 마지막에 한 번만 쓴다

## 결과 (진행 중, 검증 세트)
| 실험 | 설명 | PR-AUC | 재현율 | 정밀도 | FN | FP |
|---|---|---|---|---|---|---|
| baseline_linear | 백본 고정, 마지막 층만 학습, 임계값 0.5 | 0.8012 | 0.7195 | 0.7704 | 3,352 | 2,563 |

전체 실험 기록: `experiments.csv`, 단계별 판단 근거: `NOTES.md`

## 실행
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install scikit-learn pandas matplotlib pillow tqdm grad-cam kaggle
kaggle datasets download -d paultimothymooney/breast-histopathology-images -p data\raw
tar -xf data\raw\breast-histopathology-images.zip -C data\raw --exclude "IDC_regular_ps50_idx5/*"
python src\explore.py; python src\split.py; python src\cache.py
python src\train.py --name baseline_linear --freeze --epochs 3
```
환경: Windows, RTX 4060 Ti 8GB, Python 3.12, PyTorch 2.15 (CUDA 13.2)

## 한계
연구용 연습 프로젝트다. 진료에 쓰려면 외부 기관 데이터 검증, 의료진 검토, 허가 절차가 필요하며 최종 판단은 사람이 한다.
