# Zapptain America - Song Identifier

An audio fingerprinting app, built on the spectrogram /
constellation map / hash matching pipeline.

## What it does

- **Single clip mode**: upload one query clip and see the spectrogram, the
  constellation map (detected peaks), the offset histogram, and the predicted
  song.
- **Batch mode**: upload several query clips at once and download
  `results.csv` with exactly two columns, `filename` and `prediction`.

## Repository layout

```
fingerprint.py       core fingerprinting pipeline (spectrogram, peaks, hashes, matching)
build_database.py    one time script: fingerprints every song in a folder and saves database.pkl
database.pkl         pre built fingerprint database (committed so the app starts instantly)
app.py               the Streamlit app
requirements.txt     Python dependencies
packages.txt         system dependency (ffmpeg, needed by librosa to decode mp3)
```

## Running locally

```bash
pip install -r requirements.txt
```

If you don't already have `database.pkl`, build it once from your song folder:

```bash
python build_database.py "/path/to/your/song/database/folder"
```

This writes `database.pkl` in the current directory.

To run the app:

```bash
streamlit run app.py
```

## Notes

- If you add or change songs in the database, re run `build_database.py` and
  commit the updated `database.pkl`.
- `fingerprint.py` is shared between `build_database.py` and `app.py`, so the
  matching logic used when building the database and when matching a query is
  guaranteed to stay in sync.