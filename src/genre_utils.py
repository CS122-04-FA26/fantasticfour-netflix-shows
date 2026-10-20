"""Helper functions for loading, cleaning, and analyzing Netflix genres.

Used by notebooks/Project_Assignment_02.ipynb and by the final program.
"""

from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "netflix_titles.csv"

# Netflix uses different labels for the same genre on movies and TV shows
# (e.g. "Romantic Movies" vs. "Romantic TV Shows"). This dictionary maps every
# raw label to one base genre so movies and shows can be compared fairly.
# A value of None means the label is too generic to be useful and is dropped.
GENRE_MAP = {
    "Dramas": "Drama",
    "TV Dramas": "Drama",
    "Comedies": "Comedy",
    "TV Comedies": "Comedy",
    "Romantic Movies": "Romance",
    "Romantic TV Shows": "Romance",
    "Action & Adventure": "Action & Adventure",
    "TV Action & Adventure": "Action & Adventure",
    "Thrillers": "Thriller",
    "TV Thrillers": "Thriller",
    "Horror Movies": "Horror",
    "TV Horror": "Horror",
    "Sci-Fi & Fantasy": "Sci-Fi & Fantasy",
    "TV Sci-Fi & Fantasy": "Sci-Fi & Fantasy",
    "Documentaries": "Documentary",
    "Docuseries": "Documentary",
    "Children & Family Movies": "Kids & Family",
    "Kids' TV": "Kids & Family",
    "Crime TV Shows": "Crime",
    "TV Mysteries": "Mystery",
    "International Movies": "International",
    "International TV Shows": "International",
    "Anime Features": "Anime",
    "Anime Series": "Anime",
    "Stand-Up Comedy": "Stand-Up Comedy",
    "Stand-Up Comedy & Talk Shows": "Stand-Up Comedy",
    "Classic Movies": "Classic & Cult",
    "Cult Movies": "Classic & Cult",
    "Classic & Cult TV": "Classic & Cult",
    "Independent Movies": "Independent",
    "Music & Musicals": "Music & Musicals",
    "Reality TV": "Reality",
    "Sports Movies": "Sports",
    "Science & Nature TV": "Science & Nature",
    "LGBTQ Movies": "LGBTQ",
    "Faith & Spirituality": "Faith & Spirituality",
    "Teen TV Shows": "Teen",
    "British TV Shows": "British",
    "Spanish-Language TV Shows": "Spanish-Language",
    "Korean TV Shows": "Korean",
    "Movies": None,
    "TV Shows": None,
}


def load_netflix_data(path=DATA_PATH):
    """Load the raw Netflix dataset from the data folder."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download steps."
        )
    return pd.read_csv(path)


def split_genres(listed_in):
    """Turn 'Dramas, Romantic Movies' into ['Dramas', 'Romantic Movies']."""
    if pd.isna(listed_in):
        return []
    return [genre.strip() for genre in listed_in.split(",") if genre.strip()]


def normalize_genres(raw_genres):
    """Map raw Netflix labels to base genres, dropping duplicates and generic labels.

    Example: ['TV Dramas', 'Romantic TV Shows'] -> ['Drama', 'Romance']
    Unknown labels are kept as-is so no data is silently lost.
    """
    base_genres = []
    for genre in raw_genres:
        base = GENRE_MAP.get(genre, genre)
        if base is not None and base not in base_genres:
            base_genres.append(base)
    return base_genres


def clean_netflix_data(df):
    """Return a cleaned copy of the dataset, ready for analysis."""
    df = df.copy()

    # 3 rows have their duration ("74 min") typed into the rating column by
    # mistake, with duration left empty. Move the value back where it belongs.
    misplaced = df["rating"].str.contains("min", na=False)
    df.loc[misplaced, "duration"] = df.loc[misplaced, "rating"]
    df.loc[misplaced, "rating"] = pd.NA

    # Fill missing text fields with a clear label instead of leaving NaN.
    for column in ["director", "cast", "country", "rating"]:
        df[column] = df[column].fillna("Unknown")

    # Dates look like "September 25, 2021" (some have extra spaces).
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y")
    df["year_added"] = df["date_added"].dt.year

    # Split "90 min" / "2 Seasons" into a number and a unit.
    df["duration_value"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["duration_unit"] = df["duration"].str.contains("min", na=False).map(
        {True: "minutes", False: "seasons"}
    )

    # Genre lists: raw labels and the normalized base genres.
    df["genres_raw"] = df["listed_in"].apply(split_genres)
    df["genres"] = df["genres_raw"].apply(normalize_genres)
    df["num_genres"] = df["genres"].apply(len)
    return df


def count_genres(genre_lists):
    """Count how many titles include each genre."""
    counts = Counter()
    for genres in genre_lists:
        counts.update(genres)
    return counts


def count_genre_pairs(genre_lists, exclude=()):
    """Count how often each pair of genres appears together on one title.

    Pairs are sorted alphabetically so ('Comedy', 'Drama') and
    ('Drama', 'Comedy') count as the same pair. Genres in `exclude`
    are ignored, which lets us leave out very broad tags.
    """
    pair_counts = Counter()
    for genres in genre_lists:
        kept = sorted(g for g in genres if g not in exclude)
        for pair in combinations(kept, 2):
            pair_counts[pair] += 1
    return pair_counts


def partner_genres(genre_lists, genre, exclude=()):
    """For one genre, count which other genres it is paired with most."""
    partners = Counter()
    for genres in genre_lists:
        if genre in genres:
            partners.update(g for g in genres if g != genre and g not in exclude)
    return partners


def co_occurrence_matrix(genre_lists, genres):
    """Build a square table: cell [A, B] = number of titles tagged with both A and B."""
    matrix = pd.DataFrame(0, index=genres, columns=genres)
    for title_genres in genre_lists:
        present = [g for g in genres if g in title_genres]
        for a in present:
            for b in present:
                if a != b:
                    matrix.loc[a, b] += 1
    return matrix


def recommend_titles(df, preferred_genres, content_type=None, top_n=10):
    """Recommend titles that match the most preferred genres.

    Score = how many of the user's preferred genres a title has.
    Ties are broken by newest release year.
    """
    preferred = set(preferred_genres)
    candidates = df if content_type is None else df[df["type"] == content_type]
    scores = candidates["genres"].apply(lambda genres: len(preferred & set(genres)))
    result = candidates.assign(match_score=scores)
    result = result[result["match_score"] > 0]
    result = result.sort_values(["match_score", "release_year"], ascending=[False, False])
    columns = ["title", "type", "release_year", "rating", "listed_in", "match_score"]
    return result[columns].head(top_n).reset_index(drop=True)
