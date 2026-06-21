import sys
import pickle
from fingerprint import build_song_database

def main():
    if len(sys.argv) != 2:
        print("Usage: python build_database.py /path/to/song_database_folder")
        sys.exit(1)

    songs_dir = sys.argv[1]
    print(f"Building database from songs in: {songs_dir}\n")

    database, catalog = build_song_database(songs_dir, verbose=True)

    with open("database.pkl", "wb") as f:
        pickle.dump(database, f)
    with open("catalog.pkl", "wb") as f:
        pickle.dump(catalog, f)

    print(f"\nSaved database.pkl ({len(database)} unique hashes) " f"and catalog.pkl ({len(catalog)} songs).")

if __name__ == "__main__":
    main()