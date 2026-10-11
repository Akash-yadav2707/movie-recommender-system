# 🎬 Movie Recommendation System

A hybrid movie recommender that suggests movies based on user preferences and viewing history by finding similarities between movies and user interests. Includes a Streamlit web app and a command-line interface.

**Tools:** Python · Pandas · NumPy · Scikit-learn · SciPy · Cosine Similarity · Streamlit

![App screenshot](screenshots/app.png)

## Features
- **Find by genre / text:** type or pick genres (e.g. *action romance thriller*, *scary*, *funny*) and get the best-matching movies.
- **Similar movies:** pick a movie and get the most similar ones.
- **Personalised recommendations:** builds a taste profile from a user's rating history and recommends unseen movies.
- **Cold start:** new users get popular, well-rated movies.
- **Evaluation:** hit-rate@k (hide a user's favourite movie, check whether the model brings it back).

## How it works
1. **Content-based filtering:** movie genres are converted to TF-IDF vectors; cosine similarity measures how alike two movies are.
2. **Collaborative filtering:** item-item cosine similarity on mean-centred user ratings (users who liked X also liked Y).
3. **Hybrid score:** `score = α · content + (1 − α) · collaborative`, with a Bayesian-weighted rating so movies with very few votes don't dominate.
4. **User profile:** a user's rated movies, weighted by how much they liked each one, are combined to score every unseen movie.

Similarities are computed on demand (row by row), so it runs comfortably on the full MovieLens-small dataset (~9,700 movies, 100k ratings).

## Setup
```bash
git clone https://github.com/Akash-yadav2707/movie-recommender-system.git
cd movie-recommender-system

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
```

Download **ml-latest-small** from https://grouplens.org/datasets/movielens/latest/ and extract the folder next to `app.py`. Without it, the app falls back to a small built-in sample dataset.

## Run
**Web app**
```bash
streamlit run app.py
```

**Command line**
```bash
python movie_recommender.py --data ml-latest-small --ask
python movie_recommender.py --data ml-latest-small --user 10 --movie "Toy Story"
```

## Project structure
```
├── movie_recommender.py   # recommendation model + CLI
├── app.py                 # Streamlit web app
├── requirements.txt
└── ml-latest-small/       # MovieLens data (download separately)
```

## Dataset
[MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research (ml-latest-small). Not included in this repository.

## Future improvements
- Add movie tags / overviews to the TF-IDF features
- Matrix factorisation (SVD) for collaborative filtering
- Train/test split with precision@k and recall@k on MovieLens

## Author
**Akash Yadav** — [GitHub](https://github.com/Akash-yadav2707) · [LinkedIn](https://linkedin.com/in/akash-yadav-166981297)