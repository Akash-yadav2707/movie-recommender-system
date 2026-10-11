"""Streamlit demo for the Movie Recommendation System.
Run:  streamlit run app.py
"""
import os
import streamlit as st
from movie_recommender import MovieRecommender, load_movielens, load_sample

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading data and building model...")
def get_model(data_path):
    try:
        movies, ratings = load_movielens(data_path)
    except (SystemExit, Exception):
        movies, ratings = load_sample()
    return MovieRecommender(movies, ratings)


st.title("🎬 Movie Recommendation System")
st.caption("Python · Pandas · NumPy · Scikit-learn · Cosine Similarity")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("MovieLens folder (optional)", value="ml-latest-small",
                              help="Folder with movies.csv and ratings.csv. "
                                   "If not found, built-in sample data is used.")
    top_n = st.slider("Number of recommendations", 3, 15, 5)

rec = get_model(data_path.strip())
using_real = len(rec.movies) > 100
st.sidebar.success(f"{len(rec.movies):,} movies · {rec.ratings['userId'].nunique():,} users"
                   + ("" if using_real else " (sample data)"))

tab1, tab2, tab3 = st.tabs(["🔎 Find by genre", "🎞️ Similar movies", "👤 For a user"])

# ---- Tab 1: genre / text search -------------------------------------------
with tab1:
    st.subheader("Kaisi movie chahiye?")
    genres = sorted(rec.vocab)
    picked = st.multiselect("Genres chuno", [g.title() for g in genres],
                            default=["Action", "Romance", "Thriller"]
                            if {"action", "romance", "thriller"} <= rec.vocab else [])
    text = st.text_input("...ya likho (jaise: action romance thriller, scary, funny)")
    query = " ".join(picked) + " " + text
    if st.button("Movies dikhao", type="primary"):
        res, wanted = rec.recommend_by_query(query, top_n)
        if res is None:
            st.warning("Genre samajh nahi aaya. Koi genre chuno ya likho.")
        else:
            st.write("Genres: **" + ", ".join(sorted(wanted)) + "**")
            st.dataframe(res, hide_index=True)

# ---- Tab 2: similar movies ------------------------------------------------
with tab2:
    st.subheader("Is movie jaisi aur movies")
    choice = st.selectbox("Movie chuno", sorted(rec.movies["title"]))
    if st.button("Similar movies dikhao"):
        st.dataframe(rec.similar_movies(choice, top_n),
                     hide_index=True)

# ---- Tab 3: personalised --------------------------------------------------
with tab3:
    st.subheader("User ki viewing history ke hisaab se")
    uid = st.selectbox("User ID", sorted(rec.ratings["userId"].unique()))
    hist = rec.ratings[rec.ratings["userId"] == uid].merge(
        rec.movies[["movieId", "title"]], on="movieId").sort_values("rating", ascending=False)
    with st.expander("Is user ne kya dekha / rate kiya"):
        st.dataframe(hist[["title", "rating"]].head(20),
                     hide_index=True)
    if st.button("Recommend karo"):
        st.dataframe(rec.recommend_for_user(uid, top_n),
                     hide_index=True)