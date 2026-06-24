import os
import numpy as np
import librosa
import tempfile
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import Counter

SAMPLE_RATE = 22050
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".aiff")


def load_audio(path_or_buffer, sr=SAMPLE_RATE):
    
    try:
        y, _ = librosa.load(path_or_buffer, sr=sr, mono=True)
        return y
    except Exception:
        if isinstance(path_or_buffer, (str, os.PathLike)):
            raise
        path_or_buffer.seek(0)
        suffix = os.path.splitext(getattr(path_or_buffer, "name", ""))[1] or ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(path_or_buffer.read())
            tmp_path = tmp.name
        try:
            y, _ = librosa.load(tmp_path, sr=sr, mono=True)
            return y
        finally:
            os.remove(tmp_path)


def compute_spectrogram(y, sr=SAMPLE_RATE, nperseg=1024, noverlap=512, window="hann"):
    
    if len(y) < nperseg:
        nperseg = max(len(y), 2)
        noverlap = nperseg // 2
    f, t, Sxx = spectrogram(y, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Sxx


def spectrogram_db(Sxx):
    
    return 10 * np.log10(Sxx + 1e-10)


def find_peaks_constellation(Sxx_db, neighborhood=(15, 15), rel_threshold_db=25):
    
    local_max = (maximum_filter(Sxx_db, size=neighborhood) == Sxx_db)
    threshold = Sxx_db.max() - rel_threshold_db
    peak_mask = local_max & (Sxx_db > threshold)

    freq_idx, time_idx = np.where(peak_mask)
    peaks = sorted(zip(time_idx, freq_idx))
    return peaks


def build_fingerprints(peaks, fan_out=5, max_time_delta=50):
    
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
    
    f, t, Sxx = compute_spectrogram(y, sr)
    Sxx_db = spectrogram_db(Sxx)
    peaks = find_peaks_constellation(Sxx_db)
    fingerprints = build_fingerprints(peaks)
    return f, t, Sxx_db, peaks, fingerprints


def build_song_database(songs_dir, sr=SAMPLE_RATE, verbose=True):
    
    database = {}
    catalog = {}
    song_files = sorted(f for f in os.listdir(songs_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS))

    for filename in song_files:
        path = os.path.join(songs_dir, filename)
        y = load_audio(path, sr=sr)
        _, _, _, peaks, fingerprints = fingerprint_audio(y, sr)

        for h, t1 in fingerprints:
            database.setdefault(h, []).append((filename, t1))
        catalog[filename] = {"peaks": len(peaks), "hashes": len(fingerprints)}

        if verbose:
            print(f"  fingerprinted {filename}: {len(peaks)} peaks -> {len(fingerprints)} hashes")

    if verbose:
        print(f"\nDatabase built: {len(database)} unique hashes across {len(song_files)} songs")
    return database, catalog

def match_fingerprints(fingerprints, database):
    
    votes = Counter()
    for h, t1_query in fingerprints:
        if h in database:
            for song_name, t1_db in database[h]:
                offset = t1_db - t1_query
                votes[(song_name, offset)] += 1
    return votes.most_common()


def aggregate_by_song(votes):
    
    best_per_song = {}
    for (song_name, offset), count in votes:
        if count > best_per_song.get(song_name, 0):
            best_per_song[song_name] = count
    return sorted(best_per_song.items(), key=lambda kv: kv[1], reverse=True)


def match_query(query_audio, database, sr=SAMPLE_RATE):
    
    _, _, _, _, fingerprints = fingerprint_audio(query_audio, sr)
    return match_fingerprints(fingerprints, database)


def predict_song(query_audio, database, sr=SAMPLE_RATE):
    
    votes = match_query(query_audio, database, sr=sr)
    if not votes:
        return None, votes
    best_song_name = votes[0][0][0]
    prediction = os.path.splitext(best_song_name)[0]
    return prediction, votes

def best_offset_for_song(votes, song_name):
    
    best_offset, best_count = None, 0
    for (name, offset), count in votes:
        if name == song_name and count > best_count:
            best_offset, best_count = offset, count
    return best_offset, best_count