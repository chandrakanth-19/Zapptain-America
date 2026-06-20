"""
app.py

Q3B - "Zapp tain America": a Streamlit app that wraps the Q3A fingerprinting
pipeline into a usable song identifier.

Two modes (selected from the sidebar):
  - Single-clip mode: upload one query clip; shows the spectrogram, the
    constellation (peaks), the offset histogram, and the predicted song.
  - Batch mode: upload a set of query clips; runs all of them and produces
    results.csv with exactly two columns: filename, prediction
    (both WITHOUT file extension), as required by the assignment spec.
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from fingerprint import (
    SAMPLE_RATE,
    load_audio,
    compute_spectrogram,
    spectrogram_db,
    find_peaks_constellation,
    build_fingerprints,
    match_fingerprints,
)

DATABASE_PATH = "database.pkl"


# ----------------------------------------------------------------------
# Database loading (cached so it only loads once per app session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_database():
    if not os.path.exists(DATABASE_PATH):
        st.error(
            f"Could not find '{DATABASE_PATH}'. Run build_database.py locally "
            "first and commit the resulting database.pkl to the repo."
        )
        st.stop()
    with open(DATABASE_PATH, "rb") as f:
        return pickle.load(f)


def predict_from_audio(y, database, sr=SAMPLE_RATE):
    """
    Runs the full pipeline on a loaded waveform and returns everything needed
    for both the UI (plots) and the CSV (prediction).
    """
    f, t, Sxx = compute_spectrogram(y, sr)
    Sxx_db = spectrogram_db(Sxx)
    peaks = find_peaks_constellation(Sxx_db)
    fingerprints = build_fingerprints(peaks)
    votes = match_fingerprints(fingerprints, database)

    if votes:
        best_song_name = votes[0][0][0]
        prediction = os.path.splitext(best_song_name)[0]
    else:
        prediction = None

    return {
        "f": f, "t": t, "Sxx_db": Sxx_db,
        "peaks": peaks, "votes": votes,
        "prediction": prediction,
    }


# ----------------------------------------------------------------------
# Plot helpers (Streamlit-friendly: return a matplotlib Figure)
# ----------------------------------------------------------------------
def fig_spectrogram(f, t, Sxx_db, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(t, f, Sxx_db, shading="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    return fig


def fig_constellation(f, t, Sxx_db, peaks, title="Constellation map"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(t, f, Sxx_db, shading="auto", cmap="gray_r")
    if peaks:
        peak_t = [t[ti] for ti, fi in peaks]
        peak_f = [f[fi] for ti, fi in peaks]
        ax.scatter(peak_t, peak_f, color="red", s=18, marker="o",
                   facecolors="none", linewidths=1.1, label="Peaks")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_offset_histogram(votes, title="Time-offset histogram"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if not votes:
        ax.set_title("No matches found")
        return fig
    top_song = votes[0][0][0]
    song_votes = sorted((offset, count) for (name, offset), count in votes if name == top_song)
    offsets = [o for o, c in song_votes]
    counts = [c for o, c in song_votes]
    ax.stem(offsets, counts, basefmt=" ")
    ax.set_title(f"{title}\nBest match: {os.path.splitext(top_song)[0]}")
    ax.set_xlabel("Time offset (frames)")
    ax.set_ylabel("Votes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Zapp tain America - Song Identifier", page_icon="🎵", layout="wide")
st.title("🎵 Zapp tain America")
st.caption("A small Shazam-style song identifier, built on spectrogram fingerprinting.")

database = load_database()
st.sidebar.success(f"Database loaded: {len(database):,} unique hashes")

mode = st.sidebar.radio("Mode", ["Single-clip mode", "Batch mode"])

# ============================================================
# SINGLE-CLIP MODE
# ============================================================
if mode == "Single-clip mode":
    st.header("Single-clip mode")
    st.write("Upload one query clip to identify it, with the intermediate steps shown.")

    uploaded = st.file_uploader("Upload a query clip (mp3 or wav)", type=["mp3", "wav"])

    if uploaded is not None:
        with st.spinner("Loading audio..."):
            y = load_audio(uploaded, sr=SAMPLE_RATE)
        st.audio(uploaded)

        with st.spinner("Fingerprinting and matching..."):
            result = predict_from_audio(y, database)

        st.subheader("Result")
        if result["prediction"] is not None:
            st.success(f"**Predicted song:** {result['prediction']}")
            top_votes = result["votes"][0][1]
            st.write(f"Top match votes: **{top_votes}**")
        else:
            st.warning("No match found in the database.")

        st.subheader("Intermediate steps")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_spectrogram(result["f"], result["t"], result["Sxx_db"],
                                       title="Spectrogram"))
        with col2:
            st.pyplot(fig_constellation(result["f"], result["t"], result["Sxx_db"],
                                         result["peaks"], title="Constellation (peaks)"))

        st.pyplot(fig_offset_histogram(result["votes"], title="Offset histogram"))

        with st.expander("Top 5 raw matches"):
            for (song, offset), count in result["votes"][:5]:
                st.write(f"{os.path.splitext(song)[0]} | offset={offset} | votes={count}")

# ============================================================
# BATCH MODE
# ============================================================
else:
    st.header("Batch mode")
    st.write(
        "Upload a set of query clips. The app fingerprints each one and writes "
        "**results.csv** with exactly two columns: `filename, prediction` "
        "(both without file extension), matching the required submission format."
    )

    uploaded_files = st.file_uploader(
        "Upload query clips (mp3 or wav)", type=["mp3", "wav"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button(f"Run batch identification on {len(uploaded_files)} file(s)"):
            rows = []
            progress = st.progress(0.0)
            status = st.empty()

            for i, uploaded in enumerate(uploaded_files):
                status.write(f"Processing: {uploaded.name}")
                query_name_no_ext = os.path.splitext(uploaded.name)[0]
                try:
                    y = load_audio(uploaded, sr=SAMPLE_RATE)
                    result = predict_from_audio(y, database)
                    prediction = result["prediction"] if result["prediction"] is not None else ""
                except Exception as e:
                    st.warning(f"Could not process '{uploaded.name}': {e}")
                    prediction = ""

                rows.append({"filename": query_name_no_ext, "prediction": prediction})
                progress.progress((i + 1) / len(uploaded_files))

            status.write("Done.")
            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])

            st.subheader("Results")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )
