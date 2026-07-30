"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/beat_tracker.py

Descrption: The main entry point function for the beat tracker. This can be
imported as follows:

>>> from beat_tracking_tcn.beat_tracker import beatTracker

Then it can be invoked like so:

>>> beats, downbeats = beatTracker(path_to_audio_file)
"""
import os
import pickle
import numpy as np
from madmom.features import DBNBeatTrackingProcessor
import torch
import librosa
# from beat_tracking_tcn.models.beat_net_tcn import BeatNet
from beat_tracking_tcn.models.offlinetcn import BeatNet
# from beat_tracking_tcn.models.onlinetcn import BeatNet
# from beat_tracking_tcn.models.tcntrans import BeatNet
# from beat_tracking_tcn.models.transformer_only import BeatNet
# from beat_tracking_tcn.models.beat_net_tcn import BeatNet
# from beat_tracking_tcn.models.onlinetctempo import BeatNet
# from beat_tracking_tcn.models.transformer_only import BeatNet
# from beat_tracking_tcn.models.offline_tcn_trans import BeatNetFusionOffline
from beat_tracking_tcn.utils.spectrograms import create_spectrogram,\
                                                 trim_spectrogram


def _remap_weight_norm_keys(state_dict):
    """
    Checkpoints saved under torch >= 2.1 store weight-normed convs as
    `...parametrizations.weight.original0/original1`, while the models here use
    the deprecated `torch.nn.utils.weight_norm`, which expects `weight_g` and
    `weight_v`. Rename so old model code can load new checkpoints.
    """
    renames = {
        '.parametrizations.weight.original0': '.weight_g',
        '.parametrizations.weight.original1': '.weight_v',
    }
    out = {}
    for key, value in state_dict.items():
        for new_suffix, old_suffix in renames.items():
            if key.endswith(new_suffix):
                key = key[:-len(new_suffix)] + old_suffix
                break
        out[key] = value
    return out


def load_checkpoint(model, checkpoint_file):
    """
    Restores a model to a given checkpoint, but loads directly to CPU, allowing
    model to be run on non-CUDA devices.
    """
    state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Retry with the parametrizations -> weight_g/weight_v renaming.
        model.load_state_dict(_remap_weight_norm_keys(state_dict))


# Some important constants that don't need to be command line params
FFT_SIZE = 2048
HOP_LENGTH_IN_SECONDS = 0.05
SR = 22050
HOP_LENGTH_IN_SAMPLES = np.int64(SR * HOP_LENGTH_IN_SECONDS)
N_MELS = 81



# Paths to checkpoints distributed with the beat tracker. It's possible to
# call the below functions with custom checkpoints also.
DEFAULT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_checkpoint.torch')
DEFAULT_DOWNBEAT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_downbeat_checkpoint.torch')


# Prepare the models
model = BeatNet()
model.eval()
downbeat_model = BeatNet(downbeats=True)
downbeat_model.eval()

# # Prepare the post-processing dynamic Bayesian networks, courtesy of madmom.
# dbn = DBNBeatTrackingProcessor(
#     min_bpm=5,
#     max_bpm=25,
#     transition_lambda = 100,
#     fps= 10,
#     online=True)

from beat_tracking_tcn.utils.particle_filtering_cascade import particle_filter_cascade
dbn_pf = particle_filter_cascade(beats_per_bar=[], fps= (SR / HOP_LENGTH_IN_SAMPLES), plot=[], mode='offline', min_bpm=20, max_bpm=30, transition_lambda=100)

dbn = DBNBeatTrackingProcessor(
    min_bpm=20,
    max_bpm=30,
    transition_lambda=100,
    fps= (SR / HOP_LENGTH_IN_SAMPLES),
    online=True)
downbeat_dbn = DBNBeatTrackingProcessor(
    min_bpm=20,
    max_bpm=40,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),
    online=True)


import scipy
from scipy.signal import find_peaks


def beat_activations_from_spectrogram(
    spectrogram,
    checkpoint_file=None,
    downbeats=True):
    """
    Given a spectrogram, use the TCN model to compute a beat activation
    function.
    """

    # Load the appropriate checkpoint
    if checkpoint_file is not None:
        load_checkpoint(
            downbeat_model if downbeats else model,
            checkpoint_file)
    else:
        load_checkpoint(
            downbeat_model if downbeats else model,
            DEFAULT_DOWNBEAT_CHECKPOINT_PATH
                if downbeats else DEFAULT_CHECKPOINT_PATH)
        

    # Speed up computation by skipping torch's autograd
    with torch.no_grad():
        # Convert to torch tensor if necessary
        if type(spectrogram) is not torch.Tensor:
            spectrogram_tensor = torch.from_numpy(spectrogram)\
                                    .unsqueeze(0)\
                                    .unsqueeze(0)\
                                    .float()

        else:
            # Otherwise use the spectrogram as-is
            spectrogram_tensor = spectrogram.unsqueeze(0)\
                                    .float()
            
        # print(spectrogram_tensor.shape)
        rtrn = model(spectrogram_tensor)
        # rp = [1,2,3]
        rtrn = rtrn.numpy()
        # rp = rp.numpy()

        # Forward the spectrogram through the model. Note there are no size
        # restrictions here, as the model is fully convolutional. 
        return downbeat_model(spectrogram_tensor).numpy() if downbeats\
               else rtrn
    
def predict_beats_from_spectrogram(
        spectrogram,
        checkpoint_file=None,
        downbeats=True,
        min_bpm=40,
        max_bpm=200,
        transition_lambda=50,
        hop=None,
        online=False
   ):
    """
    Given a spectrogram, predict a list of beat times using the TCN model and
    a DBN post-processor.

    hop:    spectrogram hop length in seconds. Must match the hop the
            spectrograms were generated with, or every beat time is scaled by
            the ratio between the two. Defaults to HOP_LENGTH_IN_SECONDS.
    online: decode causally with madmom's forward algorithm instead of Viterbi.
            The results in dbpapertasks/ predating 2025-11 were produced this
            way, via DBNBeatTrackingProcessor(online=True).process().
    """
    fps = SR / np.int64(SR * (hop if hop is not None else HOP_LENGTH_IN_SECONDS))
    raw_activations = beat_activations_from_spectrogram(
        spectrogram,
        checkpoint_file,
        downbeats
    ).squeeze()

    # Perform independent post-processing for downbeats
    if downbeats:
        beat_activations = raw_activations[0]
        downbeat_activations = raw_activations[1]

        dbn.reset()
        dbn(min_bpm, max_bpm)
        predicted_beats = dbn.process_offline(beat_activations.squeeze())

        downbeat_dbn.reset()
        downbeat_dbn(min_bpm, max_bpm)
        predicted_downbeats = downbeat_dbn.process_offline(downbeat_activations.squeeze())

        return predicted_beats, predicted_downbeats
    else:
        beat_activations = raw_activations

        beat_dbn = DBNBeatTrackingProcessor(
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            transition_lambda=transition_lambda,
            fps=fps,
            online=online)

        activations = beat_activations.squeeze()
        predicted_beats = beat_dbn.process_online(activations) if online \
            else beat_dbn.process_offline(activations)

        return predicted_beats


def beatTracker(input_file, checkpoint_file=None, downbeats=True):
    """
    Our main entry point — load an audio file, create a spectrogram and predict
    a list of beat times from it.
    """    
    mag_spectrogram = create_spectrogram(
            input_file,
            FFT_SIZE,
            HOP_LENGTH_IN_SECONDS,
            N_MELS).T
    
    return predict_beats_from_spectrogram(
        mag_spectrogram,
        checkpoint_file,
        downbeats)



