import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv("https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat",
                         sep="::", engine="python",
                         names=["movie_id", "title", "genres"],
                         encoding="latin-1")
    movies["genres"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["title"] = movies["title"].str.strip()
    return movies

@st.cache_data
def build_model(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, movies, cosine_sim, n=5):
    matches = movies[movies["title"].str.contains(title, case=False, na=False)]
    if matches.empty:
        return None, None, None
    idx = matches.index[0]
    matched_title = movies.loc[idx, "title"]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    movie_indices = [s[0] for s in sim_scores]
    scores = [round(s[1], 3) for s in sim_scores]
    result = movies.loc[movie_indices, "title"].reset_index(drop=True)
    return result, scores, matched_title

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommendation System")
st.markdown("Enter a movie you like and get 5 similar recommendations!")

movies = load_data()
cosine_sim = build_model(movies)

movie_input = st.text_input("Type a movie name:", placeholder="e.g. Toy Story, Inception, Avatar")

if movie_input:
    result = get_recommendations(movie_input, movies, cosine_sim)
    if result[0] is None:
        st.error("Movie not found! Try a different title.")
    else:
        titles, scores, matched = result
        st.success(f"Showing recommendations based on: **{matched}**")
        st.subheader("Top 5 Recommendations")
        for i, (title, score) in enumerate(zip(titles, scores), 1):
            st.markdown(f"**{i}. {title}**")
            st.progress(float(score))
            st.caption(f"Similarity score: {score}")