#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# ### Algorithm:
# 1. HPSS => Harmonics, Percussives
# 1. Percussives => peak pick => noise generator
# 1. Harmonics => CQT
# 1. CQT => Bass range => peak pick => triangle generator
#   * bass: pick the peak note within [24, 60]
# 1. CQT => Treble range => peak pick => harmony => pulse generator
#   * mids: 2 peaks within [60, 96]

import argparse
import sys
import os

import librosa
import numpy as np
import scipy.signal
import functools

import warnings
warnings.filterwarnings("ignore")

# MAGIC NUMBERS
sr = 22050
MIDI_MIN = 12
MIDI_MAX = 120
fmin, fmax = librosa.midi_to_hz([MIDI_MIN, MIDI_MAX])
n_fft = 2048
hop_length = 512

TREBLE_MIN = 48 - MIDI_MIN
TREBLE_MAX = 96 - MIDI_MIN
BASS_MIN = 24 - MIDI_MIN
BASS_MAX = TREBLE_MIN - 12


def triangle(*args, **kwargs):
    '''Synthesize a triangle wave'''
    v = scipy.signal.sawtooth(*args, **kwargs)

    return 2 * np.abs(v) - 1.


def nes_triangle(*args, **kwargs):
    '''Synthesize a quantized NES triangle'''

    # http://wiki.nesdev.com/w/index.php/APU_Triangle
    # NES triangle is quantized to 16 values
    w = triangle(*args, **kwargs)

    qw = w - np.mod(w, 2./15)

    return qw


def noise(seq):
    '''Synthesize binary noise'''
    v = np.random.randint(0, 2, size=(len(seq),))

    return 2 * v - 1.


def quantize_values(X, v=15):
    '''Quantize values to at most v discrete values'''

    X = X - X.min()
    X = X / X.max()
    X = X - np.mod(X, 1./v)
    return X


def synthesize(beats, piano_roll, fmin=0, bins_per_octave=12,
               tuning=0.0, wave=None, n=None):
    '''Synthesize a weighted piano roll'''

    # Quantize the piano roll
    sr = 22050

    piano_roll = quantize_values(piano_roll)

    if wave is None:
        wave = functools.partial(scipy.signal.square, duty=0.5)

    bins_per_semi = bins_per_octave/12
    first_bin = bins_per_semi/2

    frequencies = librosa.cqt_frequencies(n_bins=piano_roll.shape[0],
                                          fmin=fmin,
                                          bins_per_octave=bins_per_octave,
                                          tuning=tuning)

    beats -= beats[0]

    if n is None:
        n = beats[-1] + 0.5 * sr

    beats = librosa.util.fix_frames(beats, x_min=0, x_max=n)
    beat_intervals = librosa.util.frame(beats, frame_length=2, hop_length=1).T

    output = np.zeros(n)

    correction = 2.0 ** (tuning / bins_per_octave)
    stream = correction * 2.0 * np.pi * np.arange(len(output)) / sr

    active_bins = piano_roll.sum(axis=1) > 0

    for n, freq in enumerate(frequencies):
        if not active_bins[n * bins_per_semi + first_bin]:
            continue

        my_f = freq * stream

        sine = wave(my_f)

        # Align beat timings to zero crossings of sine
        zc_mask = librosa.zero_crossings(sine)

        beat_f = match_zc(beat_intervals, zc_mask, freq * correction, sr)

        # Mask out this frequency wherever it's inactive
        for m, (start, end) in enumerate(beat_f):
            sine[start:end] *= piano_roll[n*bins_per_semi + first_bin, m]

        output += sine

    output = librosa.util.normalize(output)
    return output, sr


def match_zc(queries, zc_mask, my_f, sr):

    # For each query, bound define a search range
    window = int(np.ceil(sr / my_f))

    output = np.empty(queries.size, dtype=int)

    for i, q in enumerate(queries.ravel()):
        s = np.maximum(0, q - window)
        t = np.minimum(len(zc_mask), q + window)

        vals = s + np.flatnonzero(zc_mask[s:t])
        output[i] = vals[np.argmin(np.abs(q - vals))]

    return output.reshape(queries.shape)


def peakgram(C, max_peaks=1, note_search=8):
    '''Compute spectrogram column-wise peaks subject to constraints'''

    mask = np.zeros_like(C)

    for t in range(C.shape[1]):
        if t == 0:
            col = C[:, t]
        else:
            col = np.min(C[:, t]) * np.ones(C.shape[0])

            # Find peaks in the previous column
            # zero out anything outside of +- note_search from a peak
            for v in np.argwhere(mask[:, t-1]):
                r_min = max(0, v-note_search)
                r_max = min(col.shape[0], v+note_search+1)
                col[r_min:r_max] = C[r_min:r_max, t]

        # Local normalization
        z = col.max()
        if z > 0:
            col = col / z

        # Don't look for 2nds or 11ths
        # Compute max over an octave range, +- 3 semitones
        peaks = librosa.util.peak_pick(col, 3, 3, 6, 6, 1./sum(col), 3)

        if len(peaks) == 0:
            continue

        # Order the peaks by loudness
        idx = np.argsort(col[peaks])[::-1]
        peaks = peaks[idx]

        # Clip to max_peaks
        peaks = peaks[:min(len(peaks), max_peaks)]

        # If there are any peaks, pick them out
        mask[peaks, t] = 1

    return mask


def process_audio(*args, **kwargs):
    '''load the audio, do feature extraction'''

    y, sr = librosa.load(*args, **kwargs)

    # Get the harmonic and percussive components
    y_harm, y_perc = librosa.effects.hpss(y)

    # compute CQT
    cq = librosa.cqt(y_harm, sr, fmin=fmin, n_bins=108, hop_length=hop_length)

    # Trim to match cq and P shape
    P = np.abs(librosa.stft(y_perc, n_fft=n_fft, hop_length=hop_length))

    duration = min(P.shape[1], cq.shape[1])
    P = librosa.util.fix_length(P, duration, axis=1)
    cq = librosa.util.fix_length(cq, duration, axis=1)

    return y, cq, P


def get_wav(cq, nmin=60, nmax=120, width=9, max_peaks=1, wave=None, n=None):
    
    # Slice down to the bass range
    cq = cq[nmin:nmax]
    
    cq_weighted = librosa.logamplitude(cq**2, ref_power=np.max, top_db=40.0)
    
    # Pick peaks at each time
    mask = peakgram(cq_weighted, max_peaks=max_peaks)
    
    # Smooth in time
    mask = librosa.util.medfilt(mask, kernel_size=(1, width))

    wav = synthesize(librosa.frames_to_samples(np.arange(cq.shape[1]), 
                                                hop_length=hop_length), 
                             mask * cq**(1./3), 
                             fmin=librosa.midi_to_hz(nmin + MIDI_MIN), 
                             bins_per_octave=12,
                             wave=wave,
                             n=n)[0]
    
    return wav


def get_drum_wav(P, width=9, n=None):

    # Compute volume shaper
    v = np.mean(P / P.max(), axis=0, keepdims=True)
    v = librosa.util.medfilt(v, kernel_size=(1, width))
    v = librosa.util.normalize(v)

    wav = synthesize(librosa.frames_to_samples(np.arange(v.shape[1]),
                                               hop_length=hop_length),
                     v,
                     fmin=librosa.midi_to_hz(0),
                     bins_per_octave=12,
                     wave=noise,
                     n=n)[0]

    return wav


def process_arguments(args):

    parser = argparse.ArgumentParser(description='Auto Chip Tune')

    parser.add_argument('input_file',
                        action='store',
                        help='Path to the input audio file')

    parser.add_argument('output_file',
                        action='store',
                        help='Path to store the generated chiptune')

    parser.add_argument('-s', '--stereo',
                        action='store_true',
                        default=False,
                        help='Mix original and synthesized tracks in stereo')

    return vars(parser.parse_args(args))


def autochip(input_file=None, output_file=None, stereo=False):

    print 'Processing {:s}'.format(os.path.basename(input_file))
    y, cq, P = process_audio(input_file)

    print 'Synthesizing squares...'
    y_treb = get_wav(cq,
                     nmin=TREBLE_MIN,
                     nmax=TREBLE_MAX,
                     width=7,
                     max_peaks=2,
                     n=len(y))

    print 'Synthesizing triangles...'
    y_bass = get_wav(cq,
                     nmin=BASS_MIN,
                     nmax=BASS_MAX,
                     width=7,
                     wave=nes_triangle,
                     max_peaks=1,
                     n=len(y))

    print 'Synthesizing drums...'
    y_drum = get_drum_wav(P, width=3, n=len(y))

    print 'Mixing... '
    y_chip = librosa.util.normalize(0.25 * y_treb +
                                    0.25 * y_bass +
                                    0.1 * y_drum)

    if stereo:
        y_out = np.vstack([librosa.util.normalize(y), y_chip])
    else:
        y_out = y_chip

    librosa.output.write_wav(output_file, y_out, sr)
    print 'Done.'


if __name__ == '__main__':
    arguments = process_arguments(sys.argv[1:])
    autochip(**arguments)
