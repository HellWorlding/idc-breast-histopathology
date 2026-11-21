#!/usr/bin/env bash
# CPU 서버 대기열: 1~6단계를 하나씩 순서대로 돌린다. 각 단계는 끝난 부분을 건너뛰므로 끊기면 다시 실행하면 된다.
# 실행: tmux new-session -d -s idc2 -x 200 -y 50 'bash scripts/cpu_queue.sh'
# 한 단계가 실패해도 다음 단계로 넘어간다 (마지막 줄에 단계별 상태가 남는다).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
LOG=outputs/logs/cpu_queue.log
mkdir -p outputs/logs
STATUS=""

run() {
  local tag=$1; shift
  echo "=== [$tag] START $(date '+%F %T') ===" | tee -a "$LOG"
  QTAG=$tag "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  local st=DONE
  [ "$rc" -ne 0 ] && st="FAILED(rc=$rc)"
  STATUS="$STATUS $tag:$st"
  echo "=== [$tag] $st $(date '+%F %T') | 지금까지:$STATUS ===" | tee -a "$LOG"
}

run S1 $PY src/cpu_models.py --stage basic
run S2 $PY src/cpu_models.py --stage tree
run S3 $PY src/small_cnn.py
run S4 $PY src/cv_groupkfold.py
run S5 $PY src/xai.py
run S6 $PY src/bootstrap_ci.py
echo "=== QUEUE END $(date '+%F %T') |$STATUS ===" | tee -a "$LOG"
echo "[끝남] 아무 키나 누르면 닫힘"
read -r -n 1
