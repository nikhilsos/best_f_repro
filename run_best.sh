#!/bin/bash
# Reproduce the highest F-measure per rhythm pattern using the bundled code and
# checkpoints in this folder. Nothing outside this folder is read except the
# spectrograms and beat annotations under $DATA.
#
#   bash best_f_repro/run_best.sh                 # all 7 patterns
#   DRY_RUN=1 bash best_f_repro/run_best.sh       # print commands only
#   OUT=... bash best_f_repro/run_best.sh
#
# Settings are the ones the grid search found: lambda 100, Viterbi decode,
# 0.1s hop, f_measure_threshold 1.5. The threshold is a loose tolerance —
# ~50% of the inter-beat interval at 20 BPM. See README.md.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-/home/nikhil/projects/dbtracker/data/combined_data/test}
OUT=${OUT:-$HERE/results_best.txt}
SCRIPT=$HERE/code/evaluate_model.py

# pattern:spec_dir:label_dir:min_bpm:max_bpm:checkpoint
# jinyangjo is the only pattern where vocals-separated specs won.
runs=(
    "jajinmori:$DATA/jajinmori_total_spectrograms_2048_0.1:$DATA/beat_annotations/jajinmori:20:30:offline_tcn"
    "jinyangjo:$DATA/separated/jinyanjo_total_test_separated/vocal_specs_jinyangzo_total:$DATA/beat_annotations/jinyangjo:5:15:offline_tcn_vocals_only"
    "jungjungmori:$DATA/jungjungmori_total_spectrograms_2048_0.1:$DATA/beat_annotations/jungjungmori:5:30:offline_tcn"
    "jungmori:$DATA/jungmori_total_spectrograms_2048_0.1:$DATA/beat_annotations/jungmori:5:15:offline_tcn"
    "hweemori:$DATA/hweemori_total_spectrograms_2048_0.1:$DATA/beat_annotations/hweemori:40:60:offline_tcn"
    "utmori:$DATA/utmori_total_spectrograms_2048_0.1:$DATA/beat_annotations/utmori:20:30:offline_tcn"
    "utzungmori:$DATA/utzungmori_total_spectrograms_2048_0.1:$DATA/beat_annotations/utzungmori:10:20:offline_tcn"
)

[ -n "${DRY_RUN:-}" ] || : > "$OUT"

for entry in "${runs[@]}"; do
    IFS=: read -r rp spec ann min_bpm max_bpm ckpt <<< "$entry"
    cmd="python $SCRIPT $spec $ann $HERE/checkpoints/$ckpt \
--min_bpm $min_bpm --max_bpm $max_bpm --transition_lambda 100 \
--f_measure_threshold 1.5 --hop 0.1"

    if [ -n "${DRY_RUN:-}" ]; then
        echo "# $rp"
        echo "$cmd"
        continue
    fi

    echo "=== $rp ($ckpt) ===" | tee -a "$OUT"
    eval "$cmd" 2>>"$OUT" | tee -a "$OUT" | grep -o "'F-measure': [^,]*"
done

[ -n "${DRY_RUN:-}" ] || echo "Full output: $OUT"
