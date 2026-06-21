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
    aggregate_by_song,
    best_offset_for_song,
    SUPPORTED_EXTENSIONS,
)

UPLOAD_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]

DATABASE_PATH = "database.pkl"
CATALOG_PATH = "catalog.pkl"

@st.cache_resource
def load_catalog():
    if not os.path.exists(CATALOG_PATH):
        return None
    with open(CATALOG_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_database():
    if not os.path.exists(DATABASE_PATH):
        st.error(
            f"Could not find '{DATABASE_PATH}'."
        )
        st.stop()
    with open(DATABASE_PATH, "rb") as f:
        return pickle.load(f)


def predict_from_audio(y, database, sr=SAMPLE_RATE):
    f, t, Sxx = compute_spectrogram(y, sr)
    Sxx_db = spectrogram_db(Sxx)
    peaks = find_peaks_constellation(Sxx_db)
    fingerprints = build_fingerprints(peaks)
    votes = match_fingerprints(fingerprints, database)
    ranked = aggregate_by_song(votes)

    if ranked:
        best_song_name, top_score = ranked[0]
        prediction = os.path.splitext(best_song_name)[0]
        runner_up_ratio = (top_score / ranked[1][1]) if len(ranked) > 1 and ranked[1][1] > 0 else None
        best_offset, _ = best_offset_for_song(votes, best_song_name)
    else:
        best_song_name, prediction, top_score, runner_up_ratio, best_offset = None, None, 0, None, None

    return {
        "f": f, "t": t, "Sxx_db": Sxx_db, "peaks": peaks, "votes": votes, "ranked": ranked,
        "prediction": prediction, "best_song_name": best_song_name,
        "top_score": top_score, "runner_up_ratio": runner_up_ratio, "best_offset": best_offset,
    }


@st.cache_data
def get_song_fingerprint_points(_database, song_name):
    
    points = []
    for h, entries in _database.items():
        f1 = h[0]
        for name, t1 in entries:
            if name == song_name:
                points.append((t1, f1))
    return points

def fig_song_fingerprint_with_window(song_points, song_label, query_duration_frames, best_offset):
    fig, ax = plt.subplots(figsize=(10, 4))
    if song_points:
        times = [p[0] for p in song_points]
        freqs = [p[1] for p in song_points]
        ax.scatter(times, freqs, color="cyan", s=4, alpha=0.6)
    ax.axvspan(best_offset, best_offset + query_duration_frames, color="orange", alpha=0.25, label="Query window")
    ax.set_title(f"Where in the song? - {song_label}")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Freq bin")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


#plot functions
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
    ax.pcolormesh(t, f, Sxx_db, shading="auto", cmap="magma")
    if peaks:
        peak_t = [t[ti] for ti, fi in peaks]
        peak_f = [f[fi] for ti, fi in peaks]
        ax.scatter(peak_t, peak_f, color="white", s=18, marker="o",
                   facecolors="none", linewidths=1.1, label="Peaks")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_constellation_points_only(f, t, peaks, title="Constellation (points only)"):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    if peaks:
        peak_t = [t[ti] for ti, fi in peaks]
        peak_f = [f[fi] for ti, fi in peaks]
        ax.scatter(peak_t, peak_f, color="cyan", s=6)
    ax.set_title(f"{title} ({len(peaks)} peaks)", color="white")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Frequency (Hz)", color="white")
    ax.tick_params(colors="white")
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


# UI
st.set_page_config(page_title="Zapptain America : Song Identifier", page_icon="", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    header {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Zapptain America")
st.caption("Song identifier, built on spectrogram fingerprinting.")

database = load_database()
st.success(f"Database loaded: {len(database):,} unique hashes")

catalog = load_catalog()
if catalog:
    with st.expander(f"View Songs in Database ({len(catalog)})"):
        catalog_df = pd.DataFrame(
            [{"song": os.path.splitext(name)[0], "peaks": stats["peaks"], "hashes": stats["hashes"]}
             for name, stats in sorted(catalog.items())]
        )
        st.dataframe(catalog_df, use_container_width=True, hide_index=True)
else:
    st.caption("Catalog_error")


mode = st.selectbox("Select Mode", ["Single-clip mode", "Batch mode"])

# single clip mode
if mode == "Single-clip mode":
    st.header("Single-clip mode")
    st.write("Upload one query clip to identify it.")

    uploaded = st.file_uploader(f"Upload a query clip ({', '.join(UPLOAD_TYPES)})", type=UPLOAD_TYPES)

    if uploaded is not None:
        with st.spinner("Loading audio..."):
            y = load_audio(uploaded, sr=SAMPLE_RATE)
        st.audio(uploaded)

        with st.spinner("Fingerprinting and matching..."):
            result = predict_from_audio(y, database)

        st.subheader("Result")
        if result["prediction"] is not None:
            st.success(f"**Predicted song:** {result['prediction']}")
            if result["runner_up_ratio"] is not None:
                st.write(f"Cluster score **{result['top_score']}** · **{result['runner_up_ratio']:.0f}x** the runner-up")
            else:
                st.write(f"Cluster score **{result['top_score']}** (no runner-up to compare against)")

            with st.expander("Candidate scores"):
                candidates_df = pd.DataFrame(
                    [(os.path.splitext(s)[0], c) for s, c in result["ranked"][:10]],
                    columns=["song", "score"],
                )
                st.bar_chart(candidates_df.set_index("song"))
        else:
            st.warning("No match found in the database.")

        st.subheader("Intermediate steps")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(fig_spectrogram(result["f"], result["t"], result["Sxx_db"], title="Spectrogram"))
        with col2:
            st.pyplot(fig_constellation(result["f"], result["t"], result["Sxx_db"], result["peaks"], title="Constellation (overlay)"))
        with col3:
            st.pyplot(fig_constellation_points_only(result["f"], result["t"], result["peaks"], title="Constellation (points only)"))

        st.pyplot(fig_offset_histogram(result["votes"], title="Offset histogram"))

        if result["prediction"] is not None and result["best_offset"] is not None:
            st.subheader("Where in the song?")
            with st.spinner("Reconstructing full-song fingerprint..."):
                song_points = get_song_fingerprint_points(database, result["best_song_name"])
            st.pyplot(fig_song_fingerprint_with_window(
                song_points, result["prediction"], len(result["t"]), result["best_offset"]
            ))

        with st.expander("Top 5 raw matches"):
            for (song, offset), count in result["votes"][:5]:
                st.write(f"{os.path.splitext(song)[0]} | offset={offset} | votes={count}")

# batch mode
else:
    st.header("Batch mode")
    st.write(
        "Upload a set of query clips. The app fingerprints each one and gives the best match for each query clip"
    )

    uploaded_files = st.file_uploader(f"Upload query clips ({', '.join(UPLOAD_TYPES)})", type=UPLOAD_TYPES, accept_multiple_files=True)

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