"""
Movie Recommendation System
Tools: Python, Pandas, NumPy, Scikit-learn, Cosine Similarity

Run:
    python movie_recommender.py                          # built-in sample data
    python movie_recommender.py --ask                    # type: action romance thriller
    python movie_recommender.py --data ml-latest-small --ask   # real MovieLens data
    (MovieLens download: https://grouplens.org/datasets/movielens/latest/)

Approach (hybrid):
 1. Content-based  : TF-IDF on genres -> cosine similarity between movies
 2. Collaborative  : item-item cosine similarity on the user-rating matrix
 3. Personalised   : user profile built from viewing history (ratings),
                     final score = alpha * content + (1 - alpha) * collaborative
Similarities are computed on demand (row by row), so it also works on the
full MovieLens-small dataset (~9,700 movies) without using much memory.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------- data ----
def load_sample():
    movies = pd.DataFrame({
        "movieId": range(1, 27),
        "title": [
            "The Matrix", "Inception", "Interstellar", "The Dark Knight",
            "Toy Story", "Finding Nemo", "Shrek", "The Lion King",
            "Titanic", "The Notebook", "La La Land", "Pride & Prejudice",
            "Die Hard", "Mad Max: Fury Road", "John Wick", "The Terminator",
            "The Conjuring", "Get Out", "A Quiet Place", "Hereditary",
            "Mr. & Mrs. Smith", "Casino Royale", "Atomic Blonde",
            "True Lies", "Knight and Day", "Salt"],
        "genres": [
            "Action|Sci-Fi", "Action|Sci-Fi|Thriller", "Adventure|Drama|Sci-Fi",
            "Action|Crime|Drama", "Animation|Children|Comedy",
            "Animation|Children|Comedy", "Animation|Comedy|Fantasy",
            "Animation|Children|Drama", "Drama|Romance", "Drama|Romance",
            "Comedy|Drama|Musical|Romance", "Drama|Romance",
            "Action|Thriller", "Action|Adventure|Sci-Fi", "Action|Crime|Thriller",
            "Action|Sci-Fi|Thriller", "Horror|Thriller", "Horror|Mystery|Thriller",
            "Drama|Horror|Sci-Fi", "Drama|Horror|Mystery",
            "Action|Comedy|Romance|Thriller", "Action|Romance|Thriller",
            "Action|Romance|Thriller", "Action|Comedy|Romance|Thriller",
            "Action|Comedy|Romance|Thriller", "Action|Thriller"],
    })
    rng = np.random.default_rng(42)
    groups = {0: [1, 2, 3, 4, 13, 14, 15, 16, 22, 26],
              1: [5, 6, 7, 8, 9, 10, 11, 12, 21, 23, 24, 25],
              2: [17, 18, 19, 20, 4, 16]}
    rows = []
    for uid in range(1, 31):
        liked = groups[uid % 3]
        for mid in rng.choice(np.arange(1, 27), size=11, replace=False):
            base = 4.5 if mid in liked else 2.0
            rows.append((uid, int(mid),
                         float(np.clip(round(base + rng.normal(0, .6)), 1, 5))))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    return movies, ratings


def load_movielens(path):
    """Finds movies.csv / ratings.csv inside `path`, even if the zip was
    extracted into a nested folder (ml-latest-small/ml-latest-small/...)."""
    folder = None
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            if "movies.csv" in files and "ratings.csv" in files:
                folder = root
                break
    if folder is None:
        raise SystemExit(
            f"\nMovies data nahi mila: '{os.path.abspath(path)}'\n"
            "movies.csv aur ratings.csv wala folder is script (movie_recommender.py) ke "
            "saath wali directory me rakho,\nya poora path do, jaise:\n"
            '  python movie_recommender.py --data "C:\\Users\\Akash\\Downloads\\ml-latest-small" --ask\n'
            f"Abhi terminal ki current directory: {os.getcwd()}")
    movies = pd.read_csv(os.path.join(folder, "movies.csv"))
    ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))
    return movies, ratings


# --------------------------------------------------------------- model -----
class MovieRecommender:
    SYNONYMS = {"thrill": "thriller", "thrilling": "thriller", "suspense": "thriller",
                "love": "romance", "romantic": "romance", "pyaar": "romance",
                "scary": "horror", "darawani": "horror", "dar": "horror",
                "funny": "comedy", "hasi": "comedy", "scifi": "sci-fi",
                "science": "sci-fi", "kids": "children", "cartoon": "animation"}

    def __init__(self, movies, ratings, alpha=0.5):
        self.alpha = alpha
        self.movies = movies.drop_duplicates("movieId").reset_index(drop=True)
        self.movies["genres"] = self.movies["genres"].replace("(no genres listed)", "")
        self.ratings = ratings[ratings["movieId"].isin(self.movies["movieId"])]
        self.idx = {m: i for i, m in enumerate(self.movies["movieId"])}
        self.titles_lower = self.movies["title"].str.lower()
        self.vocab = {g.lower() for gs in self.movies["genres"] for g in gs.split("|") if g}
        self._build_content()
        self._build_collaborative()
        self._build_popularity()

    # 1. content-based: genres -> TF-IDF (sparse). Similarity is computed on demand.
    def _build_content(self):
        text = self.movies["genres"].str.replace("|", " ", regex=False) \
                                    .str.replace("-", "", regex=False)
        self.tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z]+")
        self.content_matrix = self.tfidf.fit_transform(text)

    # 2. collaborative: movie x user matrix of mean-centred ratings (sparse)
    def _build_collaborative(self):
        r = self.ratings
        centered = r["rating"] - r.groupby("userId")["rating"].transform("mean")
        users = {u: i for i, u in enumerate(r["userId"].unique())}
        self.item_matrix = csr_matrix(
            (centered.values,
             (r["movieId"].map(self.idx).values, r["userId"].map(users).values)),
            shape=(len(self.movies), len(users)))

    # weighted rating (Bayesian average) so movies with 1-2 votes don't dominate
    def _build_popularity(self):
        g = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        c, m = self.ratings["rating"].mean(), 10
        wr = (g["count"] * g["mean"] + m * c) / (g["count"] + m)
        self.pop = self.movies["movieId"].map(wr).fillna(c).values / 5.0

    def _sim_rows(self, rows):
        content = cosine_similarity(self.content_matrix[rows], self.content_matrix)
        collab = cosine_similarity(self.item_matrix[rows], self.item_matrix)
        return content, collab

    def find_movie(self, title):
        t = title.lower().strip()
        exact = self.titles_lower[self.titles_lower == t]
        if len(exact):
            return exact.index[0]
        part = self.titles_lower[self.titles_lower.str.contains(t, regex=False)]
        if len(part):
            return part.index[0]
        raise ValueError(f"'{title}' not found in dataset")

    def _frame(self, top, score, label="score"):
        return pd.DataFrame({"title": self.movies.loc[top, "title"].values,
                             "genres": self.movies.loc[top, "genres"].values,
                             label: score[top].round(3)})

    # --- similar movies (item-to-item) --------------------------------------
    def similar_movies(self, title, n=5):
        i = self.find_movie(title)
        content, collab = self._sim_rows([i])
        sim = (self.alpha * content[0] + (1 - self.alpha) * collab[0]) * (0.9 + 0.1 * self.pop)
        sim[i] = -1
        return self._frame(np.argsort(sim)[::-1][:n], sim)

    # --- personalised recommendations (viewing history) ---------------------
    def recommend_for_user(self, user_id, n=5):
        hist = self.ratings[self.ratings["userId"] == user_id]
        if hist.empty:
            return self.popular(n)
        pos = hist["movieId"].map(self.idx).values
        w = (hist["rating"] - hist["rating"].mean()).values
        if np.allclose(w, 0):
            w = np.ones_like(w)
        content, collab = self._sim_rows(pos)
        denom = np.abs(w).sum() + 1e-9
        score = (self.alpha * (content * w[:, None]).sum(0)
                 + (1 - self.alpha) * (collab * w[:, None]).sum(0)) / denom
        score = score * (0.9 + 0.1 * self.pop)
        score[pos] = -np.inf
        return self._frame(np.argsort(score)[::-1][:n], score)

    # --- natural-language / genre search ------------------------------------
    def recommend_by_query(self, query, n=5):
        words = [w.strip(",.!?").lower() for w in query.replace("|", " ").split()]
        words = [self.SYNONYMS.get(w, w) for w in words]
        wanted = {w for w in words if w in self.vocab}
        if not wanted:
            return None, wanted
        q = " ".join(w.replace("-", "") for w in wanted)
        sim = cosine_similarity(self.tfidf.transform([q]), self.content_matrix)[0]
        has_all = self.movies["genres"].apply(
            lambda g: wanted <= {x.lower() for x in g.split("|")}).values
        score = sim + 1.0 * has_all + 0.1 * self.pop
        top = np.argsort(score)[::-1][:n]
        out = self._frame(top, sim, "match")
        out["match"] = (sim[top] * 100).round(0).astype(int).astype(str) + "%"
        return out, wanted

    def popular(self, n=5):
        top = np.argsort(self.pop)[::-1][:n]
        return self._frame(top, self.pop * 5, "avg_rating")

    # --- evaluation: hide each user's best-rated movie, check if recovered --
    def hit_rate(self, k=5):
        hits, users = 0, 0
        for uid, h in self.ratings.groupby("userId"):
            if len(h) < 4:
                continue
            held = h.sort_values("rating", ascending=False).iloc[0]
            train = h[h["movieId"] != held["movieId"]]
            tmp = MovieRecommender(self.movies, pd.concat(
                [self.ratings[self.ratings["userId"] != uid], train]), self.alpha)
            rec = tmp.recommend_for_user(uid, k)
            title = self.movies.loc[self.idx[held["movieId"]], "title"]
            hits += int(title in rec["title"].values)
            users += 1
        return hits / max(users, 1)


# ------------------------------------------------------------------ CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to MovieLens folder (movies.csv, ratings.csv)")
    ap.add_argument("--user", type=int, default=1)
    ap.add_argument("--movie", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--ask", action="store_true",
                    help="interactive mode: type genres like 'action romance thriller'")
    args = ap.parse_args()

    movies, ratings = load_movielens(args.data) if args.data else load_sample()
    rec = MovieRecommender(movies, ratings)

    if args.ask:
        print("\nMovie finder ready. Example: action romance thriller   (type 'exit' to quit)")
        while True:
            q = input("\nKaisi movie chahiye? > ").strip()
            if q.lower() in {"exit", "quit", "q", ""}:
                break
            res, wanted = rec.recommend_by_query(q, args.top)
            if res is None:
                print("Genre samajh nahi aaya. Try: action, romance, thriller, comedy, horror, sci-fi, drama")
            else:
                print(f"Genres: {', '.join(sorted(wanted))}\n{res.to_string(index=False)}")
        raise SystemExit

    title = args.movie or movies["title"].iloc[0]
    print(f"\nMovies similar to '{title}':\n{rec.similar_movies(title, args.top).to_string(index=False)}")
    print(f"\nTop {args.top} recommendations for user {args.user}:\n"
          f"{rec.recommend_for_user(args.user, args.top).to_string(index=False)}")
    if not args.data:
        print(f"\nHit-rate@{args.top} (held-out favourite movie): {rec.hit_rate(args.top):.2f}")