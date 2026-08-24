#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAYWORLD_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${REMIND_PYTHON_BIN:-python3}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  TASK_IDS_VALUE="$1"
  shift
  IFS=',' read -r -a TASK_IDS_ARRAY <<< "$TASK_IDS_VALUE"

  DATA_ROOT="${REMIND_DATA_ROOT:-${PLAYWORLD_ROOT}/data}"
  MAPPING_JSON="${REMIND_MAPPING_JSON:-}"
  if [[ -z "$MAPPING_JSON" ]]; then
    CATEGORY="${TASK_IDS_ARRAY[0]%%[0-9]*}"
    CATEGORY_UPPER="$(printf '%s' "$CATEGORY" | tr '[:lower:]' '[:upper:]')"
    case "$CATEGORY_UPPER" in
      GC) DATA_SPLIT="gc" ;;
      IF) DATA_SPLIT="if" ;;
      OE) DATA_SPLIT="${REMIND_OE_SPLIT:-insight}" ;;
      *)
        echo "Cannot infer dataset split from task ID: ${TASK_IDS_ARRAY[0]}" >&2
        exit 2
        ;;
    esac
    MAPPING_JSON="${DATA_ROOT}/${DATA_SPLIT}/data.json"
  fi

  if [[ ! -f "$MAPPING_JSON" ]]; then
    echo "ReMind mapping JSON not found: $MAPPING_JSON" >&2
    exit 2
  fi

  exec "$PYTHON_BIN" "$SCRIPT_DIR/player.py" \
    --mapping-json "$MAPPING_JSON" \
    --images-dir "$DATA_ROOT" \
    --output-dir "${REMIND_OUT_ROOT:-${PLAYWORLD_ROOT}/outputs/remind}" \
    --tasks "${TASK_IDS_ARRAY[@]}" \
    "$@"
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: ./run_task.sh GC002[,GC004] [--dry-run]" >&2
  echo "Or: ./run_task.sh --mapping-json /path/to/data.json --tasks GC002" >&2
  exit 2
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/player.py" "$@"
