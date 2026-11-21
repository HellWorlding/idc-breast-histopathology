# IDC 유방 병리 패치 분류: 암을 놓치는 비용을 고려한 임계값 선택

유방 병리 슬라이드의 50×50 패치를 보고 침윤성 유관암(IDC)이 있는지 분류한다.
정확도보다 **암을 놓치는 오류(FN)**를 줄이는 데 초점을 두고, 데이터 누수를 막는 분할과 판단 근거 확인까지 포함한다.

## 데이터
- Kaggle [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) (CC0-1.0). 이 저장소에는 데이터를 넣지 않았다
- 환자 279명, 패치 277,524개, 양성 28.39% (직접 셈, `src/explore.py`)
- 원본 zip 안에 같은 패치의 사본 폴더(`IDC_regular_ps50_idx5`)가 있어 제외했다. 섞이면 같은 패치가 학습과 테스트에 동시에 들어간다

### 데이터 구조와 기초 통계 (`src/describe.py`, 전체 결과는 `outputs/table_summary.md`)
입력은 이미지이고, 표 데이터는 패치 목록 하나다. 결측값은 없다.

| 열 | 자료형 | 고유값 수 | 설명 |
|---|---|---|---|
| patient | str | 279 | 환자 ID. 분할 단위 |
| label | int | 2 | 0 = IDC 음성, 1 = IDC 양성 (폴더 이름 0/1) |
| path | str | 277,524 | `<환자>_idx5_x<X>_y<Y>_class<C>.png` |
| split | str | 3 | train / val / test (환자 단위) |
| x, y | int | (파생) | 파일 이름에서 꺼낸 슬라이드 안 좌표(픽셀). x 1~4,001, y 1~2,951 |

- 파일 이름 검사: 파싱 실패, 이름 속 class와 폴더 라벨 불일치, 이름 속 환자와 폴더 불일치, 같은 환자 안 좌표 중복, 50px 격자 밖 좌표가 모두 0건이다
- 50×50이 아닌 가장자리 패치 2,302개(0.8%)는 흰색으로 채워 50×50으로 맞췄다
  - 가장 작은 조각은 50×1이다. 가장자리 패치의 양성 비율은 train 1.0%, val 0.6%로 전체(28%)보다 훨씬 낮고, 면적 25% 미만 조각(train 339개)에는 양성이 없다. "흰 패딩이 크면 음성"이라는 지름길이 데이터에 있다는 뜻이라 배경 패치 성능을 따로 확인한다

환자 단위 통계 (279명)

| | 패치 수 | 양성 패치 수 | 양성 비율 |
|---|---|---|---|
| 평균 | 994.7 | 282.4 | 0.308 |
| 최소 | 63 | 10 | 0.010 |
| 중앙값 | 967 | 198 | 0.267 |
| 최대 | 2,395 | 1,347 | 0.904 |

세트별 요약

| 세트 | 환자 | 패치 | 양성 | 양성 비율 |
|---|---|---|---|---|
| train | 195 | 195,224 | 54,604 | 0.280 |
| val | 42 | 40,999 | 11,951 | 0.292 |
| test | 42 | 41,301 | 12,231 | 0.296 |

픽셀 통계 (train만 사용, 0~255)

| 클래스 | 패치 | R 평균 | G 평균 | B 평균 | 거의 흰 픽셀 비율 |
|---|---|---|---|---|---|
| IDC− | 140,620 | 216.4 | 172.1 | 193.6 | 28.4% |
| IDC+ | 54,604 | 186.4 | 138.1 | 172.4 | 9.6% |

- 양성 패치가 전체적으로 더 어둡고(세포핵이 빽빽함), 흰 배경·지방이 적다
- 즉 색만으로도 어느 정도 구분이 된다. 모델이 조직 구조가 아니라 색에 기대는지 색 증강과 Grad-CAM으로 확인할 계획이다

## 방법
1. **환자 단위 분할** 70/15/15, 환자별 양성 비율로 층화, 세트 간 겹치는 환자 0명 검사 (`src/split.py`)
   - 분할 결과(환자 ID 목록)를 `data/splits/*_patients.csv`로 저장소에 올려, 다른 컴퓨터에서도 `--from-lists`로 똑같은 분할을 재현한다
2. **평가 기준을 먼저 고정**: FN 비용 = FP 비용의 k배 (k = 1, 3, 5, 7, 10), 주 지표는 기대 비용, 보조 지표는 재현율, 정밀도, F2, PR-AUC
   - k를 하나로 정하지 않은 이유: 판정 단위가 패치라 패치 하나를 놓치는 비용은 환자를 놓치는 것보다 작고, 실제 k는 의료진과 정할 값이라 구간으로 비교한다
3. **전처리** (`outputs/figures/preprocessing.png`): 가장자리 패치 흰색 패딩 → 50→224 확대 → ImageNet 평균/표준편차 정규화
   - 정규화에 ImageNet 통계를 써서 우리 데이터 어느 세트에서도 통계를 계산하지 않는다 (정규화 누수 없음)
4. **CNN 전이학습** (`src/train.py`): ImageNet 사전학습 ResNet18
   - 베이스라인: 백본 고정, 마지막 층만 학습
   - 이후: 백본 전체 학습 + 증강 비교. 증강은 GPU에서 배치 단위로 한다
     - flip: 좌우/상하 뒤집기, 90도 회전 (병리 패치는 방향에 의미가 없음)
     - flipcolor: 위 + 밝기·대비·채도·색조 작은 무작위 변화 (`outputs/figures/color_aug.png`). 양성이 더 어둡다는 색 지름길과 염색 차이를 줄이려는 목적
   - 불균형 대응: 양성 손실 가중치 pos_weight = 2.58 (train 음성/양성 비율)
   - 에폭 선택은 val PR-AUC (임계값과 무관한 지표)
5. **비교용 비-딥러닝 모델** (`src/tabular.py`, CPU 서버에서 병렬 실행)
   - 고정 ResNet18 512차원 피처 → 로지스틱 회귀 / RandomForest / XGBoost
   - 색 통계 31개 피처(RGB 평균·표준편차, 흰 픽셀 비율, 히스토그램) → XGBoost: 색만으로 얼마나 맞히는지 = 색 지름길의 크기
6. **임계값 선택**: val 예측으로 k별 기대 비용 최소 임계값을 고르고 이론값 1/(1+k)와 비교한다
7. 임계값과 하이퍼파라미터는 검증 세트로만 정하고, **테스트 세트는 모든 선택이 끝난 뒤 한 번만** 쓴다

## 결과 (진행 중, 검증 세트)
| 실험 | 설명 | PR-AUC | ROC-AUC | 재현율 | 정밀도 | FN | FP |
|---|---|---|---|---|---|---|---|
| baseline_linear | 백본 고정, 마지막 층만 학습, 3에폭 | 0.8012 | 0.9074 | 0.7195 | 0.7704 | 3,352 | 2,563 |
| ft_flip | 백본 전체 학습, 뒤집기/회전, 5에폭 중 4 | 0.8900 | 0.9470 | 0.7922 | 0.8201 | 2,483 | 2,077 |
| ft_color | + 색 증강 | 0.9005 | 0.9523 | 0.8187 | 0.8230 | 2,167 | 2,104 |
| ft_posw | + 양성 가중치 2.58 | 0.9011 | 0.9525 | 0.8983 | 0.7376 | 1,215 | 3,820 |

(재현율·정밀도·FN·FP는 임계값 0.5 기준. 세 실험 모두 val PR-AUC가 가장 높은 4에폭을 골랐다)

- 패치당 기대 비용(임계값 0.5, k=1 / 5 / 10): baseline 0.144 / 0.471 / 0.880 → ft_color 0.104 / 0.316 / 0.580 → ft_posw 0.123 / 0.241 / 0.390. 전부 "음성"이라고 답하면 0.292 / 1.458 / 2.915
- 백본 전체 학습이 가장 큰 개선이었고, 색 증강은 PR-AUC와 FN을 조금 더 개선했다
- 양성 가중치는 PR-AUC(순위 매기는 능력)를 거의 바꾸지 않고 확률을 양성 쪽으로 옮겨, 임계값 0.5에서 FN을 2,167 → 1,215로 줄이는 대신 FP를 늘렸다. 임계값을 k별로 다시 고르면 비슷한 효과를 낼 수 있어 6단계에서 비교한다
- 진행 중: 다른 모델(efficientnet_b0, densenet121, swin_t) 비교, 비-딥러닝 비교(CPU 서버), k별 임계값 선택

전체 실험 기록: `experiments.csv`, `outputs/tabular_results.csv`. 단계별 판단 근거: `NOTES.md`

## 파일 구성
| 파일 | 하는 일 |
|---|---|
| `src/explore.py` | 환자·패치 수 집계, 이미지 크기 확인, 샘플 그림 |
| `src/split.py` | 환자 단위 층화 분할 (`--from-lists`: 저장된 환자 목록으로 재현) |
| `src/cache.py` | 세트별 uint8 배열로 묶기 (가장자리 흰색 패딩) |
| `src/describe.py` | 표 구조(df.info), 파일 이름 검사, 좌표·환자·픽셀 통계 |
| `src/preprocess_viz.py` | 전처리 단계별 그림 |
| `src/common.py` | 정규화, 증강, 모델, 예측, 평가 지표(기대 비용 포함) |
| `src/train.py` | CNN 학습. 에폭마다 `outputs/logs/<이름>.json` 갱신, 결과는 `experiments.csv` |
| `src/tabular.py` | 고정 피처/색 피처 + 로지스틱 회귀·RF·XGBoost. 중단 후 재실행하면 이어서 진행 |
| `scripts/server_setup.sh` | 리눅스 CPU 서버 준비 (패키지, 데이터, 분할, 캐시) |

## 실행
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install scikit-learn pandas matplotlib pillow tqdm kaggle six xgboost tabulate grad-cam
kaggle datasets download -d paultimothymooney/breast-histopathology-images -p data\raw
tar -xf data\raw\breast-histopathology-images.zip -C data\raw --exclude "IDC_regular_ps50_idx5/*"
python src\explore.py; python src\split.py --from-lists; python src\cache.py; python src\describe.py
python -u src\train.py --name baseline_linear --freeze --epochs 3
python -u src\train.py --name ft_flip --epochs 5 --lr 1e-4 --aug flip
python -u src\train.py --name ft_color --epochs 5 --lr 1e-4 --aug flipcolor
python -u src\train.py --name ft_posw --epochs 5 --lr 1e-4 --aug flipcolor --pos-weight 2.58
```
리눅스 CPU 서버: `bash scripts/server_setup.sh` 후 `python -u src/tabular.py` (Kaggle 토큰은 `~/.kaggle/`에 미리 둔다)

환경: Windows, RTX 4060 Ti 8GB, Python 3.12, PyTorch 2.15 (CUDA 13.2). 8GB GPU에서 224 입력으로 백본 전체를 학습할 때 학습·예측 배치는 256이 적당하다 (예측 배치 1024에서는 GPU 메모리가 넘쳐 학습이 수 배 느려졌다, `NOTES.md` 10절)

## 한계
- 연구용 연습 프로젝트다. 진료에 쓰려면 외부 기관 데이터 검증, 의료진 검토, 허가 절차가 필요하며 최종 판단은 사람이 한다.
- **환자 단위 분리를 완전히 보장할 수 없다.** 데이터 출처 설명은 여성 162명의 슬라이드라고 하지만, 환자 폴더는 279개다. 이 프로젝트는 폴더 ID를 환자로 보고 나눴는데, 한 사람의 슬라이드가 여러 폴더로 나뉘어 있다면 같은 사람의 조직이 train과 test에 함께 들어갔을 수 있다. 폴더와 실제 환자의 대응 정보가 공개되어 있지 않아 확인하지 못했고, 그 경우 성능이 실제보다 다소 높게 나올 수 있다.
