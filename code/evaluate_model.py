"""
Improved Version by ChatGPT - 2025

Script: evaluate_model.py
Description: Evaluation script for beat tracking using mir_eval and model predictions.
Author: Adapted from Ben Hayes (2020)

Reference:
[1] M. E. P. Davies, N. Degara, and M. D. Plumbley, 'Evaluation Methods for Musical Audio Beat Tracking Algorithms', 2009.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from mir_eval.beat import evaluate

# Import the bundled copy of beat_tracking_tcn, never the main project tree.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from beat_tracking_tcn.beat_tracker import predict_beats_from_spectrogram
from beat_tracking_tcn.datasets.ballroom_dataset import BallroomDataset


def parse_args():
    parser = ArgumentParser(description="Evaluate a BeatNet model")
    parser.add_argument("spectrogram_dir", type=str)
    parser.add_argument("label_dir", type=str)
    parser.add_argument("model_checkpoint", type=str)
    parser.add_argument("--downbeats", action="store_true")
    parser.add_argument("--rp", type=str, default=None, help="Rhythm Pattern mode (optional)")
    parser.add_argument("--min_bpm", type=int, default=5, help="Minimum BPM for DBN postprocessing")
    parser.add_argument("--max_bpm", type=int, default=10, help="Maximum BPM for DBN postprocessing")
    parser.add_argument("--transition_lambda", type=float, default=None, help="DBN transition lambda (overrides per-pattern default)")
    parser.add_argument("--f_measure_threshold", type=float, default=0.07,
                        help="mir_eval F-measure tolerance in seconds (default=0.07; results predating 2025-11 used 1.5)")
    parser.add_argument("--hop", type=float, default=None,
                        help="Spectrogram hop length in seconds; must match the spectrogram dir (default: beat_tracker.HOP_LENGTH_IN_SECONDS)")
    parser.add_argument("--online", action="store_true",
                        help="Decode with madmom's causal forward algorithm instead of Viterbi")
    return parser.parse_args()

def bpmselect(rp) -> tuple:
    """
    Selects the minimum and maximum BPM based on the rhythm pattern mode.
    """
    if rp == "jajinmori":
        return 20, 30
    elif rp == "jinyangjo":
        return 5, 15
    elif rp == "jungmori":
        return 5, 15
    elif rp == "jungjungmori":
        return 5, 30
    elif rp == "hweemori":
        return 40, 60
    elif rp == "utmori":
        return 20, 30
    elif rp == "utzungmori":
        return 10, 20
    else:
        raise ValueError("Invalid rhythm pattern mode specified.")

# Per-pattern DBN transition_lambda values.
# Lower lambda → more permissive tempo transitions (less phase-locked).
# Higher lambda → more rigid tempo tracking.
# Baseline lambda=50 was tuned for hweemori; patterns with precision gaps
# (too many false positives) benefit from a lower lambda.
LAMBDA_DEFAULTS = {
    "jajinmori":    75,   # sweep: F=0.0629 @ lambda=75 (0.05hop specs)
    "jinyangjo":   100,   # sweep: F=0.0756 @ lambda=100 (0.05hop specs)
    "jungmori":     75,   # sweep: F=0.2853 @ lambda=75  (0.05hop specs)
    "jungjungmori": 10,   # sweep: F=0.3054 @ lambda=10  (0.05hop specs)
    "hweemori":     50,
    "utmori":       50,
    "utzungmori":   50,
}

def lambdaselect(rp) -> float:
    """
    Returns the DBN transition_lambda for the given rhythm pattern.
    """
    if rp not in LAMBDA_DEFAULTS:
        raise ValueError(f"Unknown rhythm pattern: {rp}")
    return LAMBDA_DEFAULTS[rp]
    
def evaluate_model(model_checkpoint, spectrogram, ground_truth, downbeats=True, min_bpm=40, max_bpm=200,
                   transition_lambda=50, f_measure_threshold=0.07, hop=None, online=False):
    prediction = predict_beats_from_spectrogram(
        spectrogram, model_checkpoint, downbeats=downbeats, min_bpm=min_bpm, max_bpm=max_bpm,
        transition_lambda=transition_lambda, hop=hop, online=online)

    if downbeats:
        scores = (
            evaluate(ground_truth[0], prediction[0]),
            evaluate(ground_truth[1], prediction[1])
        )
    else:
        prediction = prediction[:len(ground_truth)]
        scores = evaluate(ground_truth, prediction, f_measure_threshold=f_measure_threshold)

    return scores, prediction

def evaluate_model_on_dataset(model_checkpoint, dataset, ground_truths, downbeats=True, min_bpm=40, max_bpm=200,
                              transition_lambda=50, f_measure_threshold=0.07, hop=None, online=False,
                              print_callback=None):
    mean_scores, mean_downbeat_scores = {}, {}
    running_scores, running_downbeat_scores = {}, {}
    correct = 0

    for i in range(len(dataset)):
        # print(f"Processing example {i + 1}/{len(dataset)}", end="\r")
        data = dataset[i]
        spectrogram = data["spectrogram"]
        # spectrogram, gt_rhythm_pattern = data["spectrogram"], data["rhythm_pattern"]
        ground_truth = ground_truths[i]
        name = dataset.get_name(i)

        scores, prediction = evaluate_model(
            model_checkpoint, spectrogram, ground_truth, downbeats, min_bpm=min_bpm, max_bpm=max_bpm,
            transition_lambda=transition_lambda, f_measure_threshold=f_measure_threshold,
            hop=hop, online=online)

        # predicted_pattern = torch.argmax(rhythm_pattern_predictions, dim=1)
        # if predicted_pattern == gt_rhythm_pattern:
        #     correct += 1

        # accuracy = correct / (i + 1)
        if downbeats:
            beat_scores, downbeat_scores = scores
            for metric in downbeat_scores:
                running_downbeat_scores[metric] = running_downbeat_scores.get(metric, 0.0) + downbeat_scores[metric]
        else:
            beat_scores = scores

        for metric in beat_scores:
            running_scores[metric] = running_scores.get(metric, 0.0) + beat_scores[metric]

        if print_callback:
            print_callback(i, running_scores)

        # # Save logs
        # save_logs(name, ground_truth, prediction)

    for metric in running_scores:
        mean_scores[metric] = running_scores[metric] / len(dataset)
    if downbeats:
        for metric in running_downbeat_scores:
            mean_downbeat_scores[metric] = running_downbeat_scores[metric] / len(dataset)

    return {
        "total_examples": len(dataset),
        "scores": mean_scores,
        "downbeat_scores": mean_downbeat_scores,
        # "rhythm_pattern_accuracy": accuracy
    }


# def save_logs(name, ground_truth, prediction):
#     gt_file = Path(f"/home/nikhil/projects/dbtracker/scripts/downbeat_logs/ground_truth/{name}.txt")
#     pred_file = Path(f"/home/nikhil/projects/dbtracker/scripts/downbeat_logs/prediction/{name}.txt")

#     gt_lines = [f"{num:.6f}\t{num:.6f}\tB" for num in ground_truth]
#     pred_lines = [f"{num:.6f}\t{num:.6f}\tB" for num in prediction]

#     gt_file.parent.mkdir(parents=True, exist_ok=True)
#     pred_file.parent.mkdir(parents=True, exist_ok=True)

#     gt_file.write_text('\n' + '\n'.join(gt_lines))
#     pred_file.write_text('\n' + '\n'.join(pred_lines))


def print_callback(i, running_scores):
    def make_metric_heading(metric):
        words = metric.split(" ")
        for idx, word in enumerate(words):
            for vowel in "aeiouAEIOU":
                words[idx] = words[idx][0] + words[idx][1:].replace(vowel, "")
        return "".join(words)

    if i == 0:
        headings = " | ".join([make_metric_heading(metric).ljust(10) for metric in running_scores])
        print(headings)

    line = " | ".join([
        f"{running_scores[metric]/(i+1):.4f}".ljust(10)
        for metric in running_scores
    ])
    print(line, end="\r")


if __name__ == "__main__":
    args = parse_args()

    dataset = BallroomDataset(
        spectrogram_dir=args.spectrogram_dir,
        label_dir=args.label_dir,
        downbeats=args.downbeats)
    
    # Echo the config: these captures get redirected to files and later nobody
    # can tell which settings produced which numbers.
    print(f"config: {vars(args)}")
    print(f"Loaded dataset with {len(dataset)} examples.")

    ground_truths = (
        tuple(zip(
            [dataset.get_ground_truth(i) for i in range(len(dataset))],
            [dataset.get_ground_truth(i, downbeats=True) for i in range(len(dataset))]))
        if args.downbeats
        else [dataset.get_ground_truth(i) for i in range(len(dataset))]
    )

    min_bpm, max_bpm = bpmselect(args.rp) if args.rp else (args.min_bpm, args.max_bpm)

    if args.transition_lambda is not None:
        transition_lambda = args.transition_lambda
    elif args.rp is not None:
        transition_lambda = lambdaselect(args.rp)
    else:
        transition_lambda = 50

    results = evaluate_model_on_dataset(
        args.model_checkpoint,
        dataset,
        ground_truths,
        downbeats=args.downbeats,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        transition_lambda=transition_lambda,
        f_measure_threshold=args.f_measure_threshold,
        hop=args.hop,
        online=args.online,
        print_callback=print_callback
    )

    print("\nEvaluation complete.")
    print(results)



# """
# Ben Hayes 2020

# ECS7006P Music Informatics

# Coursework 1: Beat Tracking

# File: scripts/evaluate_model.py

# Descrption: Evaluate a single given beat tracking model according to metrics as
#             outlined in Davies et al 2009 [1]. Provides functions for other
#             evaluation scripts to use.

# References:
#     [1] M. E. P. Davies, N. Degara, and M. D. Plumbley, ‘Evaluation Methods for
#         Musical Audio Beat Tracking Algorithms’, p. 17.
# """
# from argparse import ArgumentParser
# import sys
# import numpy as np
# # sys.path.append('/home/nikhil/TCN/')
# # print(sys.path)
# from mir_eval.beat import evaluate
# from mir_eval.beat import trim_beats
# import torch
# sys.path.append('/home/nikhil/projects/dbtracker/')
# from beat_tracking_tcn.beat_tracker import predict_beats_from_spectrogram
# from beat_tracking_tcn.datasets.ballroom_dataset import BallroomDataset
# # from beat_tracking_tcn.datasets.dataloading import BallroomDataset
# def parse_args():
#     parser = ArgumentParser(
#         description="Perform evaluation on a given BeatNet model")

#     parser.add_argument("spectrogram_dir", type=str)
#     parser.add_argument("label_dir", type=str)
#     parser.add_argument("model_checkpoint", type=str)
#     parser.add_argument("--downbeats", action="store_true")
#     parser.add_argument("--dataset", type=str)

#     return parser.parse_args()

# def evaluate_model(
#         model_checkpoint,
#         spectrogram,
#         ground_truth,
#         downbeats=True):
#     """
#     Given a model checkpoint, a single spectrogram, and the corresponding
#     ground truth, evaluate the model's performance on all beat tracking metrics
#     offered by mir_eval.beat.
#     """        

#     prediction = predict_beats_from_spectrogram(
#         spectrogram,
#         model_checkpoint,
#         downbeats=downbeats)
    
    

#     if downbeats:
#         # ground_truth[0] = trim_beats(ground_truth[0], min_beat_time=5.0)
#         # prediction[0] = trim_beats(prediction[0], min_beat_time=5.0)
#         # ground_truth[1] = trim_beats(ground_truth[1], min_beat_time=5.0)
#         # prediction[1] = trim_beats(prediction[1], min_beat_time=5.0)

#         scores = (evaluate(ground_truth[0], prediction[0], ),
#                   evaluate(ground_truth[1], prediction[1]))
#     else:

#         # ground_truth = trim_beats(ground_truth, min_beat_time=5.0)
#         # prediction = trim_beats(prediction, min_beat_time=5.0)
#         prediction = prediction[:len(ground_truth)]

#         scores = evaluate(ground_truth, prediction)

#     return scores

# def evaluate_model_on_dataset(
#         model_checkpoint,
#         dataset,
#         ground_truths,
#         downbeats=True,
#         print_callback=None):
#     """
#     Run through a whole instance of torch.utils.data.Dataset and compare the
#     model's predictions to the given ground truths.
#     """        

#     # Create dicts to store scores and histories
#     mean_scores = {}
#     mean_downbeat_scores = {}
#     running_scores = {}
#     running_downbeat_scores = {}

#     # Iterate over dataset
#     for i in range(len(dataset)):
#         # print(len(dataset))
#         #  catch index if more than trim size
#         # if name hainsworth, then only proceed 
#         # if ((dataset.get_name(i)[0].split('/')[-1]) == 'hainsworth' ): # or (dataset.get_name(i)[0].split('/')[-1]) == 'ballroom')
#             # print(dataset.get_name(i)[0].split('/')[-1])

#         spectrogram = dataset[i]["spectrogram"].unsqueeze(0)
        
#         ground_truth = ground_truths[i]


#         scores = evaluate_model(
#             model_checkpoint,
#             spectrogram,
#             ground_truth,
#             downbeats)

#         # If we're tracking downbeats, separate out the evaluation scores and
#         # process independently. Otherwise, we only need to worry about beat
#         scores
#         if downbeats:
#             beat_scores = scores[0]
#             downbeat_scores = scores[1]
#             for metric in downbeat_scores:
#                 if metric not in running_downbeat_scores:
#                     running_downbeat_scores[metric] = 0.0
                
#                 running_downbeat_scores[metric] += downbeat_scores[metric]
#         else:
#             beat_scores = scores

#         for metric in beat_scores:
#             if metric not in running_scores:
#                 running_scores[metric] = 0.0
#             running_scores[metric] += beat_scores[metric]
        
#         # Each iteration, pass our current index and our running score total
#         # to a print callback function.
#         if print_callback is not None:
#             print_callback(i, running_scores)

#         # After all iterations, calculate mean scores.
#         for metric in running_scores:
#             mean_scores[metric] = running_scores[metric] / (i + 1)        
#         if downbeats:
#             for metric in running_downbeat_scores:
#                 mean_downbeat_scores[metric] =\
#                     running_downbeat_scores[metric] / (i + 1)        

#     # Return a dictionary of helpful information
#     return {
#         "total_examples": i + 1,
#         "scores": mean_scores,
#         "downbeat_scores": mean_downbeat_scores
#     }

# if __name__ == "__main__":
#     args = parse_args()

#     spectrogram_dirs = [
#         # '/home/nikhil/TCN/beat-tracking-tcn/combined/spectrograms/ballroom',
#         #'/home/nikhil/TCN/beat-tracking-tcn/combined/spectrograms/hainsworth'
#         '/home/nikhil/TCN/beat-tracking-tcn/ballroom_specs'
#     ]
#     label_dirs = [
#         '/home/nikhil/TCN/beat-tracking-tcn/combined/labels/ballroom',
#         #'/home/nikhil/TCN/beat-tracking-tcn/combined/labels/hainsworth'
#     ]

#     def print_callback(i, running_scores):
#         """
#         Evaluation function set up such that scores are passed to a callback
#         after each iteration — this allows for printing or logging of results.
#         This function prints results to a table in real time.
#         """        
#         # print('this is being accesed or not?', i)
#         def make_metric_heading(metric):
#             # In order to fit the results on screen, let's strip all vowels,
#             # except the first one, and spaces from the given metric name.
            
#             words = metric.split(" ")
#             for i, _ in enumerate(words):
#                 for vowel in "aeiouAEIOU":
#                     words[i] = words[i][0] + words[i][1:].replace(vowel, "")
#             return "".join(words)

#         # The first iteration of the first fold, we also need to print the
#         # table headnings.
#         if i == 0:
#             line = ""
#             for metric in running_scores:
#                 metric_heading = make_metric_heading(metric)
#                 heading = " %s " % metric_heading
#                 # Pad any headings shorter than 6 characters so that we have
#                 # enough space for at least 4 decimal places.
#                 if len(metric_heading) < 6:
#                     padding_length = int((6 - len(metric_heading)) / 2)
#                     padding = " " * padding_length
#                     heading = padding + heading + padding
#                     if len(metric_heading) % 2 == 1:
#                         heading += " "
#                 heading += "|"
#                 line += heading          
#             print(line)

#         # Build a line of scores, truncating the decimal places to match the
#         # length of the given heading.
#         line = ""
#         for metric in running_scores:
#             metric_heading = make_metric_heading(metric)
#             number_length = len(metric_heading) - 2
#             line += " {1:.{0}f} |".format(
#                 4,
#                 running_scores[metric] / (i + 1))
#         # Print, overwriting the previously printed line each time.
#         print(line, end="\r")

#     # Load the dataset from the given directories
#     dataset = BallroomDataset(
#         spectrogram_dirs,
#         label_dirs,
#         downbeats=args.downbeats)
#     # file = args.dataset + '.pt'

#     # dataset = torch.load(file)

#     # Process downbeats and beats independently if necessary
#     if args.downbeats:
#         ground_truths = tuple(zip(
#             [dataset.get_ground_truth(i) for i in range(len(dataset))],
#             [dataset.get_ground_truth(i, downbeats=True)
#                 for i in range(len(dataset))]))
#     else:
#         ground_truths = [dataset.get_ground_truth(i) for i in range(len(dataset))]
#         # print(len(ground_truths))

#     # Run evaluation
#     result = evaluate_model_on_dataset(
#         args.model_checkpoint,
#         dataset,
#         ground_truths,
#         args.downbeats,
#         print_callback)
#     print('\n')
#     print(result)

#     # log results and input args in a text file
#     with open('evaluation_results.txt', 'w') as f:
        
#         f.write(str(args))
#         f.write(str(result))




