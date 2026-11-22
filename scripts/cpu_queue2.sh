#!/usr/bin/env bash
# CPU 서버 2차 대기열 (밤샘용): 단계를 하나씩 순서대로 돌리고, 단계가 성공하면 그 결과 파일을 스스로 git commit 한다.
# push는 하지 않는다. 각 단계는 끝난 부분을 건너뛰므로 끊기면 다시 실행하면 된다.
# 실행: tmux new-session -d -s idc3 -x 200 -y 50 'bash scripts/cpu_queue2.sh'
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
LOG=outputs/logs/cpu_queue2.log
mkdir -p outputs/logs
STATUS=""
AN=$(git log -1 --format=%an)
AE=$(git log -1 --format=%ae)

# run <태그> <커밋 메시지> <커밋할 파일들(공백 구분, 없는 파일은 건너뜀)> -- <명령...>
run() {
  local tag=$1 msg=$2 files=$3; shift 3; shift
  echo "=== [$tag] START $(date '+%F %T') ===" | tee -a "$LOG"
  QTAG=$tag "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  local st=DONE
  if [ "$rc" -ne 0 ]; then
    st="FAILED(rc=$rc)"
  else
    local add=()
    for f in $files; do [ -e "$f" ] && add+=("$f"); done
    if [ ${#add[@]} -gt 0 ]; then
      git add -- "${add[@]}"
      if git diff --cached --quiet; then
        echo "[$tag] 커밋할 변경 없음" | tee -a "$LOG"
      else
        git -c user.name="$AN" -c user.email="$AE" commit -q -m "$msg" \
          -m "Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" 2>&1 | tee -a "$LOG"
        echo "[$tag] commit $(git log -1 --format='%h %s')" | tee -a "$LOG"
      fi
    fi
  fi
  STATUS="$STATUS $tag:$st"
  echo "=== [$tag] $st $(date '+%F %T') | 지금까지:$STATUS ===" | tee -a "$LOG"
}

run T1 "Add calibration results (queue 2 stage 1: ECE, Brier, reliability diagrams)" \
  "outputs/calibration.csv outputs/calibration_bins.csv outputs/figures/calib_reliability_cnn.png outputs/figures/calib_reliability_tree.png" \
  -- $PY src/calib_threshold.py --part calib
run T2 "Add threshold analysis (queue 2 stage 2: cost-min thresholds, patient-split estimate)" \
  "outputs/threshold_analysis.csv outputs/threshold_split.csv" \
  -- $PY src/calib_threshold.py --part threshold
run T3 "Add per-patient recall at k=5 cost-min threshold (queue 2 stage 3)" \
  "outputs/patient_recall.csv outputs/patient_recall_summary.csv outputs/figures/calib_patient_recall_k5.png" \
  -- $PY src/calib_threshold.py --part patient
# 4단계: 색 실험. 두 실행을 합쳐 3시간(180분) 안에 들도록 실행 하나의 한도를 85분으로 둔다.
run T4a "Add small CNN grayscale-input run (queue 2 stage 4a: 30 epochs, seed 42)" \
  "outputs/smallcnn_results.csv outputs/logs/smallcnn_gray_e30_s42.json" \
  -- $PY src/small_cnn.py --name smallcnn_gray_e30_s42 --gray --epochs 30 --seed 42 --budget 85
run T4b "Add small CNN color-input run (queue 2 stage 4b: 30 epochs, seed 42)" \
  "outputs/smallcnn_results.csv outputs/logs/smallcnn_color_e30_s42.json" \
  -- $PY src/small_cnn.py --name smallcnn_color_e30_s42 --epochs 30 --seed 42 --budget 85
# 5단계: 시드 반복 (역시 실행 하나에 85분 한도)
run T5a "Add small CNN color-input run (queue 2 stage 5: seed 1)" \
  "outputs/smallcnn_results.csv outputs/logs/smallcnn_color_e30_s1.json" \
  -- $PY src/small_cnn.py --name smallcnn_color_e30_s1 --epochs 30 --seed 1 --budget 85
run T5b "Add small CNN color-input run (queue 2 stage 5: seed 2)" \
  "outputs/smallcnn_results.csv outputs/logs/smallcnn_color_e30_s2.json" \
  -- $PY src/small_cnn.py --name smallcnn_color_e30_s2 --epochs 30 --seed 2 --budget 85
echo "=== QUEUE2 END $(date '+%F %T') |$STATUS ===" | tee -a "$LOG"
echo "[끝남] 아무 키나 누르면 닫힘"
read -r -n 1
