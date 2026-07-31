"""
Predict downbeat (jangdan cycle start) timestamps for a single audio file.

    python infer.py track.mp3 --rp jungmori
    python infer.py track.mp3 --rp jungmori --out track.beats
    python infer.py track.wav --min-bpm 5 --max-bpm 15

The annotations this model was trained and evaluated against mark only cycle
starts, so the model's output *is* the downbeat track — there is no separate
beat level to collapse. The BPM ranges below are cycles per minute; the median
cycle rate measured from the test annotations sits inside each range:

    jajinmori 26.6 | jinyangjo 6.1 | jungjungmori 13.5 | jungmori 6.9
    hweemori 45.8  | utmori 22.8   | utzungmori 16.3
"""
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))

from beat_tracking_tcn.beat_tracker import predict_beats_from_spectrogram
from beat_tracking_tcn.utils.spectrograms import create_spectrogram

# Settings the reported F-measures were produced with. Changing HOP without
# retraining rescales every predicted timestamp.
FFT_SIZE = 2048
HOP = 0.1
N_MELS = 81
LAMBDA = 100

# pattern -> (min_bpm, max_bpm) in cycles per minute, from bpmselect()
RHYTHM_PATTERNS = {
    'jajinmori': (20, 30),
    'jinyangjo': (5, 15),
    'jungjungmori': (5, 30),
    'jungmori': (5, 15),
    'hweemori': (40, 60),
    'utmori': (20, 30),
    'utzungmori': (10, 20),
}

# offline_tcn is the best checkpoint for six of seven patterns; jinyangjo scored
# highest with offline_tcn_vocals_only on vocals-separated input. See README.md.
DEFAULT_CHECKPOINT = HERE / 'checkpoints/offline_tcn'
PER_PATTERN_CHECKPOINT = {'jinyangjo': HERE / 'checkpoints/offline_tcn_vocals_only'}


def infer(audio_path, min_bpm, max_bpm, checkpoint):
    spectrogram = create_spectrogram(str(audio_path), FFT_SIZE, HOP, N_MELS).T
    return predict_beats_from_spectrogram(
        spectrogram,
        str(checkpoint),
        downbeats=False,     # this checkpoint has a single output channel
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        transition_lambda=LAMBDA,
        hop=HOP)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio', type=Path, help='.mp3 or .wav')
    ap.add_argument('--rp', choices=sorted(RHYTHM_PATTERNS),
                    help='rhythm pattern; sets the cycle-rate range and checkpoint')
    ap.add_argument('--min-bpm', type=float, help='override, cycles per minute')
    ap.add_argument('--max-bpm', type=float, help='override, cycles per minute')
    ap.add_argument('--checkpoint', type=Path, help='override checkpoint')
    ap.add_argument('--out', type=Path,
                    help='write "<time>\\t1" per line (.beats format); default stdout')
    args = ap.parse_args()

    if args.rp:
        min_bpm, max_bpm = RHYTHM_PATTERNS[args.rp]
    elif args.min_bpm is not None and args.max_bpm is not None:
        min_bpm, max_bpm = args.min_bpm, args.max_bpm
    else:
        ap.error('pass --rp, or both --min-bpm and --max-bpm')

    if args.min_bpm is not None:
        min_bpm = args.min_bpm
    if args.max_bpm is not None:
        max_bpm = args.max_bpm

    checkpoint = args.checkpoint or PER_PATTERN_CHECKPOINT.get(args.rp, DEFAULT_CHECKPOINT)
    if not Path(checkpoint).exists():
        ap.error(f'checkpoint not found: {checkpoint}')
    if not args.audio.exists():
        ap.error(f'audio not found: {args.audio}')

    downbeats = infer(args.audio, min_bpm, max_bpm, checkpoint)

    if args.out:
        args.out.write_text(''.join(f'{t:.6f}\t1\n' for t in downbeats))
        print(f'{len(downbeats)} downbeats -> {args.out}')
    else:
        for t in downbeats:
            print(f'{t:.6f}')


if __name__ == '__main__':
    main()
