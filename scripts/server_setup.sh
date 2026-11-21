#!/usr/bin/env bash
# CPU 서버 준비: 패키지 → Kaggle 데이터 → 압축 해제(중복 폴더 제외) → 탐색 → 분할(환자 목록) → 캐시
# 다시 실행해도 된다: 끝난 단계는 건너뛴다 (서버 정지 대비).
# 실행: bash scripts/server_setup.sh   (저장소 루트에서, kaggle 토큰은 미리 ~/.kaggle/ 에 넣어 둘 것)
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "1. 시스템 패키지"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip unzip tmux git > /dev/null

log "2. 파이썬 가상환경 (CPU용 torch)"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy pandas scikit-learn xgboost pillow tqdm matplotlib kaggle six tabulate  # six: kaggle 2.2.5가 필요로 하지만 의존성에 빠져 있음
python -c "import torch, torchvision, xgboost, sklearn; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| xgboost', xgboost.__version__, '| sklearn', sklearn.__version__)"

log "3. Kaggle 데이터 다운로드"
ZIP=data/raw/breast-histopathology-images.zip
mkdir -p data/raw
if [ ! -f "$ZIP" ]; then
  if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ~/.kaggle/access_token ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    echo "kaggle 토큰 없음: ~/.kaggle/kaggle.json (또는 access_token)을 넣고 chmod 600 후 다시 실행"; exit 1
  fi
  kaggle datasets download -d paultimothymooney/breast-histopathology-images -p data/raw
fi
ls -l "$ZIP"

log "4. 압축 해제 (중복 폴더 IDC_regular_ps50_idx5 제외)"
if [ ! -f data/raw/.extracted ]; then
  unzip -q -o "$ZIP" -x 'IDC_regular_ps50_idx5/*' -d data/raw
  touch data/raw/.extracted
fi
n_pat=$(find data/raw -mindepth 1 -maxdepth 1 -type d ! -name IDC_regular_ps50_idx5 | wc -l)
n_png=$(find data/raw -name '*.png' -not -path '*IDC_regular_ps50_idx5*' | wc -l)
echo "환자 폴더 $n_pat (기대 279), PNG $n_png (기대 277524)"
[ "$n_pat" -eq 279 ] && [ "$n_png" -eq 277524 ] || { echo "데이터 수가 데스크톱과 다름"; exit 1; }

log "5. 탐색 (patches_index.csv)"
[ -f outputs/patches_index.csv ] || python src/explore.py > /dev/null

log "6. 분할: 저장된 환자 목록 사용 (데스크톱과 같은 분할)"
python src/split.py --from-lists > /dev/null
python - <<'EOF'
import json
r = json.load(open("data/splits/split_report.json"))
expect = {"train": (195, 195224, 54604), "val": (42, 40999, 11951), "test": (42, 41301, 12231)}
for s, (p, n, pos) in expect.items():
    got = (r[s]["patients"], r[s]["patches"], r[s]["pos"])
    assert got == (p, n, pos), f"{s}: 서버 {got} != 데스크톱 {(p, n, pos)}"
print("분할 일치: 환자/패치/양성 수가 데스크톱과 같음")
EOF

log "7. 캐시 (data/cache/*.npy)"
if [ ! -f data/cache/test_meta.csv ]; then
  python src/cache.py 2>&1 | grep -v '%|'
fi
ls -l data/cache
log "완료. 다음: python -u src/tabular.py"
