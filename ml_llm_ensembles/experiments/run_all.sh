#!/usr/bin/env bash
# Run the released experiment suite in dependency order.
#
#   synthetic  experiment-13 unit tests and fake-backend smoke run; no data
#   smoke      reduced-size execution of each supported experiment; data required
#   paper      complete paper configuration; GPU, data, and substantial time required

set -uo pipefail
cd "$(dirname "$0")/../.."

export PATH="$HOME/.local/bin:$PATH"
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PY=(uv run python)
EXP_DIR="ml_llm_ensembles/experiments"
MODE="${1:-synthetic}"
FAILED=()

run() {
  local label="$1"
  shift
  echo
  echo "===== $label ====="
  if ! "${PY[@]}" "$@"; then
    echo "FAILED: $label"
    FAILED+=("$label")
  fi
}

if [ "$MODE" = "synthetic" ]; then
  export EXPERIMENT_RESULT_SUFFIX=".synthetic"
  run "experiment 13 invariant tests" "$EXP_DIR/tests_13_synthetic.py"
  run "experiment 13 fake-backend smoke" "$EXP_DIR/13_phishing_operational.py" \
    --synthetic-smoke --models xgb modernbert_ft --ft-backend fake \
    --ft-scope minimal --ft-output-dir models/smoke/exp13-ft --n-boot 20 --no-plots
  if [ ${#FAILED[@]} -eq 0 ]; then
    echo "synthetic checks completed"
    exit 0
  fi
  exit 1
fi

if [ "$MODE" != "smoke" ] && [ "$MODE" != "paper" ]; then
  echo "usage: bash ml_llm_ensembles/experiments/run_all.sh [synthetic|smoke|paper]"
  exit 2
fi

if [ "$MODE" = "paper" ] && [ ! -f data/phishing-operational/manifest.csv ]; then
  echo "paper mode requires data/phishing-operational/manifest.csv"
  echo "See data/phishing-operational/README.md for the manifest and source prerequisites."
  exit 2
fi

if [ "$MODE" = "smoke" ]; then
  export EXPERIMENT_RESULT_SUFFIX=".smoke"
  export EXPERIMENT_LIMIT="${EXPERIMENT_LIMIT:-150}"
  FULL=0
  FT_PHISH="models/smoke/ft-phishing"
  FT_FLOWS="models/smoke/ft-flows"
  EPOCHS=(--epochs 1)
  FOLDS=(--folds 2)
  PACKET_ARGS=(--samples 200)
  AFTERIMAGE_ARGS=(--samples 400)
  CICIDS_ARGS=(--samples 200)
  CTU_ARGS=(--train-rows 2000 --tabpfn-rows 500 --bert-rows 500 \
            --expensive-test-rows 500 --llm-test-pos 30 --llm-test-neg 70 \
            --llm-train-rows 100)
  KITSUNE_ARGS=(--tabpfn-rows 500 --expensive-test-rows 500 \
                --llm-test-pos 30 --llm-test-neg 70 --llm-train-rows 100)
else
  export EXPERIMENT_RESULT_SUFFIX=""
  unset EXPERIMENT_LIMIT
  FULL=1
  FT_PHISH="models/modernbert-phishing-ft"
  FT_FLOWS="models/modernbert-mirai-flows-ft"
  EPOCHS=(--epochs 3)
  FOLDS=()
  PACKET_ARGS=()
  AFTERIMAGE_ARGS=()
  CICIDS_ARGS=()
  CTU_ARGS=()
  KITSUNE_ARGS=()
fi

CLAUDE_ARGS=()
if [ "${INCLUDE_CLAUDE_CACHE:-0}" = "1" ]; then
  CLAUDE_ARGS=(--claude)
fi

NETWORK_LLM_ARGS=(--models)
KITSUNE_LLM_ARGS=(--models)
if [ "${ALLOW_LIVE_LLM_CALLS:-0}" = "1" ]; then
  NETWORK_LLM_ARGS=(--models mistral gemma3:12b llama3.2 --allow-llm-calls)
  KITSUNE_LLM_ARGS=(--models mistral llama3.2 --allow-llm-calls)
fi

run "01 phishing fine-tune, raw" "$EXP_DIR/01_ft_phishing.py" \
  "${EPOCHS[@]}" --output-dir "$FT_PHISH"
if [ "$FULL" = 1 ]; then
  run "01 phishing fine-tune, stripped" "$EXP_DIR/01_ft_phishing.py" \
    "${EPOCHS[@]}" --strip-provenance --output-dir "${FT_PHISH}-stripped"
fi
run "02 flow fine-tune" "$EXP_DIR/02_ft_flows.py" "${EPOCHS[@]}" --output "$FT_FLOWS"

run "03 phishing router, raw" "$EXP_DIR/03_router_phishing.py" \
  --ft-dir "$FT_PHISH" "${CLAUDE_ARGS[@]}"
if [ "$FULL" = 1 ]; then
  run "03 phishing router, stripped" "$EXP_DIR/03_router_phishing.py" \
    --strip-provenance --ft-dir "${FT_PHISH}-stripped" "${CLAUDE_ARGS[@]}"
fi
run "04 flow router" "$EXP_DIR/04_router_flows.py" \
  --ft-dir "$FT_FLOWS" "${CLAUDE_ARGS[@]}"

run "05 phishing stack, raw" "$EXP_DIR/05_meta_phishing.py" "${FOLDS[@]}"
if [ "$FULL" = 1 ]; then
  run "05 phishing stack, stripped" "$EXP_DIR/05_meta_phishing.py" \
    --strip-provenance "${FOLDS[@]}"
fi
run "06 flow stack" "$EXP_DIR/06_meta_flows.py" "${FOLDS[@]}"
run "07 packet stack" "$EXP_DIR/07_meta_pcap.py" "${FOLDS[@]}" "${PACKET_ARGS[@]}"
run "08 AfterImage temporal" "$EXP_DIR/08_meta_afterimage.py" \
  --split temporal "${FOLDS[@]}" "${AFTERIMAGE_ARGS[@]}"
if [ "$FULL" = 1 ]; then
  run "08 AfterImage random reference" "$EXP_DIR/08_meta_afterimage.py" \
    --split random --allow-leaky-random "${FOLDS[@]}" "${AFTERIMAGE_ARGS[@]}"
fi
run "09 CIC-IDS2017 stack" "$EXP_DIR/09_meta_cicids.py" \
  "${FOLDS[@]}" "${CICIDS_ARGS[@]}"

run "11 CTU-13 scenario" "$EXP_DIR/11_ctu13_operational.py" \
  --split scenario "${FOLDS[@]}" "${CTU_ARGS[@]}" "${NETWORK_LLM_ARGS[@]}"
run "12 Kitsune cross-capture" "$EXP_DIR/12_kitsune_cross_capture.py" \
  --mode cross "${FOLDS[@]}" "${KITSUNE_ARGS[@]}" "${KITSUNE_LLM_ARGS[@]}"

if [ "$FULL" = 1 ]; then
  run "11 CTU-13 host-grouped" "$EXP_DIR/11_ctu13_operational.py" \
    --split host "${CTU_ARGS[@]}" --models
  run "11 CTU-13 temporal" "$EXP_DIR/11_ctu13_operational.py" \
    --split temporal "${CTU_ARGS[@]}" --models
  run "12 SYN DoS within-capture, flow-grouped" "$EXP_DIR/12_kitsune_cross_capture.py" \
    --mode within --within-split flow "${KITSUNE_ARGS[@]}" --models
  run "12 SYN DoS within-capture, temporal" "$EXP_DIR/12_kitsune_cross_capture.py" \
    --mode within --within-split temporal "${KITSUNE_ARGS[@]}" --models
  run "13 phishing operational panel" "$EXP_DIR/13_phishing_operational.py" \
    --manifest data/phishing-operational/manifest.csv \
    --models xgb bert_lr modernbert_ft --ft-scope full \
    --min-source-rows 1000 --spam-split-bulk 1.0
else
  run "13 phishing operational smoke" "$EXP_DIR/13_phishing_operational.py" \
    --synthetic-smoke --models xgb modernbert_ft --ft-backend fake \
    --ft-scope minimal --ft-output-dir models/smoke/exp13-ft --n-boot 20 --no-plots
fi

if [ ${#FAILED[@]} -eq 0 ]; then
  echo
  echo "$MODE run completed. Result JSON files are local working artifacts and are ignored by Git."
  exit 0
fi

echo
echo "$MODE run failed: ${FAILED[*]}"
exit 1
