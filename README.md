# best_f_repro — highest-F configuration, self-contained

Everything here is a **copy**. Nothing was moved out of the main project.

## Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install "numpy==1.26.4" "Cython==3.0.12"   # madmom builds against these
pip install -r requirements.txt
```

Python 3.10.12, torch 2.1.2, numpy 1.26.4 (must stay <2 — madmom's Cython
extensions are built against the 1.x ABI), madmom from git at commit
`27f032e`. See `requirements.txt` for the full pinned set.

## Run

```bash
bash best_f_repro/run_best.sh          # all 7 patterns -> results_best.txt
DRY_RUN=1 bash best_f_repro/run_best.sh   # print commands only
```

Verified 2026-07-30 — all seven values below reproduce exactly.

| pattern | F-measure | checkpoint | spectrograms |
|---|---|---|---|
| jungmori | 0.9891 | offline_tcn | jungmori_total_spectrograms_2048_0.1 |
| jungjungmori | 0.9870 | offline_tcn | jungjungmori_total_spectrograms_2048_0.1 |
| utmori | 0.9754 | offline_tcn | utmori_total_spectrograms_2048_0.1 |
| jinyangjo | 0.9752 | offline_tcn_vocals_only | vocal_specs_jinyangzo_total (vocals-separated) |
| utzungmori | 0.9716 | offline_tcn | utzungmori_total_spectrograms_2048_0.1 |
| hweemori | 0.9675 | offline_tcn | hweemori_total_spectrograms_2048_0.1 |
| jajinmori | 0.8917 | offline_tcn | jajinmori_total_spectrograms_2048_0.1 |

Settings for every run: `transition_lambda=100`, Viterbi decode
(`process_offline`), `hop=0.1`, `f_measure_threshold=1.5`, per-pattern BPM range
from `bpmselect()`.

## Inference on a new audio track

```bash
python infer.py track.mp3 --rp jungmori                 # downbeat times to stdout
python infer.py track.mp3 --rp jungmori --out track.beats
python infer.py track.wav --min-bpm 5 --max-bpm 15      # unknown rhythm pattern
```

`--rp` picks both the cycle-rate range and the checkpoint (`offline_tcn`, or
`offline_tcn_vocals_only` for jinyangjo). Output is one timestamp per line, or
`<time>\t1` lines in `.beats` format with `--out`.

**The output is the downbeat track.** The `.beats` annotations mark only jangdan
cycle starts, so the model has one output channel and predicts cycle starts
directly — there is no separate beat level. This is also why the BPM ranges are
5–60: they are *cycles* per minute, not beats. Median cycle rate measured from
the test annotations, each inside its `bpmselect()` range:

| pattern | median cycle | cycles/min | range used |
|---|---|---|---|
| hweemori | 1.31 s | 45.8 | 40–60 |
| jajinmori | 2.25 s | 26.6 | 20–30 |
| utmori | 2.63 s | 22.8 | 20–30 |
| utzungmori | 3.67 s | 16.3 | 10–20 |
| jungjungmori | 4.46 s | 13.5 | 5–30 |
| jungmori | 8.67 s | 6.9 | 5–15 |
| jinyangjo | 9.81 s | 6.1 | 5–15 |

Sanity check on `zungmori_mp3/1.mp3` (jungmori) against its annotation:
38 predicted vs 37 reference downbeats, F@1.5 = 1.000, F@0.5 = 1.000,
F@0.07 = 0.528. Spectrograms are computed inside `infer.py` at 2048 FFT /
0.1 s hop / 81 mels, matching the dirs used for the evaluation table.

## Contents

```
checkpoints/offline_tcn               md5 4621e9c8ac56148ede40d1794f1f7386
                                      copy of dbpapertasks/offline_tcn (2025-02-27)
checkpoints/offline_tcn_vocals_only   md5 4c1f3b20f3583f6514c0674e14a230aa
                                      copy of beat_tracking_tcn/checkpoints/offline_tcn_vocals_only
code/evaluate_model.py                copy of scripts/evaluate_model.py, sys.path
                                      pointed at code/ so it imports the bundled package
code/beat_tracking_tcn/               beat_tracker.py, models/offlinetcn.py,
                                      utils/{spectrograms,particle_filtering_cascade}.py,
                                      datasets/ballroom_dataset.py
run_best.sh                           the 7 winning commands
```

`beat_tracker.py` here is the current working-tree version: `HOP_LENGTH_IN_SECONDS
= 0.05`, but `run_best.sh` passes `--hop 0.1` to match the spectrogram dirs, so
the DBN runs at 10 fps. Changing the spectrogram dir without changing `--hop`
rescales every predicted beat time.

Spectrograms and annotations are **not** copied; they are read from
`/home/nikhil/projects/dbtracker/data/combined_data/test` (override with `DATA=`).

## Caveats to carry into the paper

1. **`f_measure_threshold=1.5` is a loose tolerance.** mir_eval's default is
   0.07. At 20 BPM the inter-beat interval is 3 s, so ±1.5 s accepts a beat that
   is half a period off. The same predictions score ~0.09 at 0.07. A
   tempo-relative tolerance (±17.5% IBI ≈ 0.5 s here) is the defensible middle.
2. **The checkpoint was chosen per pattern by test-set F.** That is an oracle
   upper bound, not a generalization estimate. Report it as such, or re-select on
   a validation split.
3. `offline_tcn_vocals_only` and `offline_tcn_vocals_only_0.05hop` scored
   identically on jinyangjo and utzungmori — verify with md5 whether they are
   actually two distinct models.

## Provenance

`dbpapertasks/offline_tcn` beat every other candidate on 6 of 7 patterns in a
336-evaluation grid (24 checkpoints x 14 pattern/spectrogram-dir pairs); see
`scripts/grid_search_checkpoints.py` and `dbpapertasks/grid_search.csv`. It also
beats the older `dbpapertasks/` captures on all seven patterns — hweemori by
0.38 (0.9675 vs 0.5919).
