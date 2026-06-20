"""
fingerprint.py

Core audio fingerprinting pipeline for Q3B (Zapp tain America).
This is a direct port of the Q3A notebook functions, so the app's matching
behaviour is identical to what was tested and validated in Q3A.

Pipeline: load audio -> spectrogram -> peak picking (constellation) ->
hashing (paired peaks) -> hash lookup against a database -> offset-histogram
voting -> best match.
"""

import os
import numpy as np
import librosa
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import Counter

SAMPLE_RATE = 22050


def load_audio(path_or_buffer, sr=SAMPLE_RATE):
    """Load an mp3/wav file (path or file-like buffer) as a mono waveform."""
    y, _ = librosa.load(path_or_buffer, sr=sr, mono=True)
    return y


def compute_spectrogram(y, sr=SAMPLE_RATE, nperseg=1024, noverlap=512, window="hann"):
    """
    Returns freqs (Hz), times (s), and the power spectrogram Sxx.

    If the clip is shorter than `nperseg` samples, scipy automatically shrinks
    nperseg to fit -- but noverlap must always stay smaller than nperseg, so we
    shrink noverlap to match too. This avoids a crash on very short/corrupt
    uploads instead of just letting scipy raise ValueError.
    """
    if len(y) < nperseg:
        nperseg = max(len(y), 2)
        noverlap = nperseg // 2
    f, t, Sxx = spectrogram(y, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Sxx


def spectrogram_db(Sxx):
    """Convert a power spectrogram to dB scale."""
    return 10 * np.log10(Sxx + 1e-10)


def find_peaks_constellation(Sxx_db, neighborhood=(15, 15), rel_threshold_db=25):
    """
    Find local maxima ("peaks") in a spectrogram.

    neighborhood     : (freq_size, time_size) of the local window used to test for a local max.
    rel_threshold_db : a peak must be within this many dB of the loudest point in the
                       WHOLE clip (relative threshold, not absolute -- this matters
                       because absolute loudness varies between recordings).

    Returns a list of (time_index, freq_index) tuples, sorted by time.
    """
    local_max = (maximum_filter(Sxx_db, size=neighborhood) == Sxx_db)
    threshold = Sxx_db.max() - rel_threshold_db
    peak_mask = local_max & (Sxx_db > threshold)

    freq_idx, time_idx = np.where(peak_mask)
    peaks = sorted(zip(time_idx, freq_idx))
    return peaks


def build_fingerprints(peaks, fan_out=5, max_time_delta=50):
    """
    Pair up nearby peaks into hashes: hash = (f1, f2, delta_t).

    fan_out         : how many future peaks to pair each peak with.
    max_time_delta  : only pair peaks within this many spectrogram frames of each other.

    Returns a list of (hash, t1) pairs, where t1 is the time-frame index of the
    anchor (first) peak.
    """
    fingerprints = []
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        paired = 0
        for j in range(i + 1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt <= 0:
                continue
            if dt > max_time_delta:
                break
            fingerprints.append(((f1, f2, dt), t1))
            paired += 1
            if paired >= fan_out:
                break
    return fingerprints


def fingerprint_audio(y, sr=SAMPLE_RATE):
    """Full pipeline: audio -> (f, t, Sxx_db, peaks, fingerprints)."""
    f, t, Sxx = compute_spectrogram(y, sr)
    Sxx_db = spectrogram_db(Sxx)
    peaks = find_peaks_constellation(Sxx_db)
    fingerprints = build_fingerprints(peaks)
    return f, t, Sxx_db, peaks, fingerprints


def build_song_database(songs_dir, sr=SAMPLE_RATE, verbose=True):
    """
    Walk through every mp3/wav in `songs_dir`, fingerprint it, and build a
    lookup table: hash -> list of (song_filename, anchor_time_index).
    """
    database = {}
    song_files = sorted(f for f in os.listdir(songs_dir) if f.lower().endswith((".mp3", ".wav")))

    for filename in song_files:
        path = os.path.join(songs_dir, filename)
        y = load_audio(path, sr=sr)
        _, _, _, peaks, fingerprints = fingerprint_audio(y, sr)

        for h, t1 in fingerprints:
            database.setdefault(h, []).append((filename, t1))

        if verbose:
            print(f"  fingerprinted {filename}: {len(peaks)} peaks -> {len(fingerprints)} hashes")

    if verbose:
        print(f"\nDatabase built: {len(database)} unique hashes across {len(song_files)} songs")
    return database


def match_fingerprints(fingerprints, database):
    """
    Vote on (song_name, offset) pairs given a query's fingerprints and a database.
    Returns a sorted list of ((song_name, offset), vote_count), best match first.
    """
    votes = Counter()
    for h, t1_query in fingerprints:
        if h in database:
            for song_name, t1_db in database[h]:
                offset = t1_db - t1_query
                votes[(song_name, offset)] += 1
    return votes.most_common()


def match_query(query_audio, database, sr=SAMPLE_RATE):
    """Convenience wrapper: audio -> votes, using the full fingerprinting pipeline."""
    _, _, _, _, fingerprints = fingerprint_audio(query_audio, sr)
    return match_fingerprints(fingerprints, database)


def predict_song(query_audio, database, sr=SAMPLE_RATE):
    """
    Returns the predicted song filename (without extension) and the raw votes list.
    If no match is found at all, returns (None, []).
    """
    votes = match_query(query_audio, database, sr=sr)
    if not votes:
        return None, votes
    best_song_name = votes[0][0][0]
    prediction = os.path.splitext(best_song_name)[0]
    return prediction, votes
