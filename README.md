# Zapp tain America - Song Identifier (Q3B)

A small Shazam-style audio fingerprinting app, built on the spectrogram /
constellation-map / hash-matching pipeline developed in Q3A.

## What it does

- **Single-clip mode**: upload one query clip and see the spectrogram, the
  constellation map (detected peaks), the offset histogram, and the predicted
  song.
- **Batch mode**: upload several query clips at once and download
  `results.csv` with exactly two columns, `filename` and `prediction`
  (both without file extensions).

## Repository layout

```
fingerprint.py       core fingerprinting pipeline (spectrogram, peaks, hashes, matching)
build_database.py    one-time script: fingerprints every song in a folder and saves database.pkl
database.pkl         pre-built fingerprint database (committed so the app starts instantly)
app.py               the Streamlit app itself
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

This writes `database.pkl` in the current directory. Commit it to the repo
(it's roughly 15-25 MB for ~50 songs, well within GitHub's limits).

Then run the app:

```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this folder (including `database.pkl`) to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click
   "New app".
3. Point it at your repo, branch, and `app.py` as the main file.
4. Deploy. The `packages.txt` file ensures `ffmpeg` gets installed automatically
   so mp3 decoding works on the server.

## Notes

- `database.pkl` is built once, locally, from the provided song library and
  committed to the repo -- the deployed app never re-fingerprints the song
  database itself, only the query clips you upload. This keeps startup fast
  on Streamlit Cloud's free tier.
- If you add or change songs in the database, re-run `build_database.py` and
  commit the updated `database.pkl`.
- `fingerprint.py` is shared between `build_database.py` and `app.py`, so the
  matching logic used when building the database and when matching a query is
  guaranteed to stay in sync.
