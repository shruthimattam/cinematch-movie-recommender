import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .hero {
        text-align: center;
        padding: 60px 20px 40px;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        color: #a78bfa;
        padding: 6px 18px;
        border-radius: 50px;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 64px;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 18px;
        margin-top: 16px;
        font-weight: 300;
    }

    .search-container {
        max-width: 600px;
        margin: 40px auto;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
        border-radius: 16px !important;
        color: white !important;
        font-size: 18px !important;
        padding: 20px 24px !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.3) !important;
    }

    .stTextInput label {
        color: rgba(255,255,255,0.8) !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }

    .matched-banner {
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.15));
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 16px;
        padding: 16px 24px;
        margin: 20px 0;
        color: white;
        font-size: 15px;
        text-align: center;
    }

    .matched-banner span {
        color: #a78bfa;
        font-weight: 600;
    }

    .section-title {
        color: white;
        font-size: 24px;
        font-weight: 600;
        margin: 40px 0 24px;
        text-align: center;
    }

    .movie-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }

    .movie-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    }

    .movie-card:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(167,139,250,0.3);
        transform: translateY(-2px);
    }

    .card-rank {
        font-size: 13px;
        font-weight: 600;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 600;
        color: white;
        margin-bottom: 16px;
    }

    .score-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .score-label {
        color: rgba(255,255,255,0.5);
        font-size: 13px;
        min-width: 120px;
    }

    .score-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.08);
        border-radius: 100px;
        height: 6px;
        overflow: hidden;
    }

    .score-bar-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
    }

    .score-value {
        color: #60a5fa;
        font-size: 14px;
        font-weight: 600;
        min-width: 40px;
        text-align: right;
    }

    .genre-tag {
        display: inline-block;
        background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.25);
        color: #c4b5fd;
        padding: 3px 10px;
        border-radius: 50px;
        font-size: 12px;
        margin: 4px 4px 0 0;
    }

    .stats-row {
        display: flex;
        justify-content: center;
        gap: 40px;
        padding: 30px 0;
        margin: 20px 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    .stat-item {
        text-align: center;
    }

    .stat-number {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .stat-label {
        color: rgba(255,255,255,0.4);
        font-size: 13px;
        margin-top: 4px;
    }

    .error-box {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 16px;
        padding: 20px 24px;
        color: #fca5a5;
        text-align: center;
        font-size: 15px;
    }

    .footer {
        text-align: center;
        padding: 40px 20px;
        color: rgba(255,255,255,0.2);
        font-size: 13px;
        margin-top: 60px;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    .footer span {
        color: #a78bfa;
    }

    .how-it-works {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 30px;
        margin: 40px 0;
    }

    .how-title {
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
    }

    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 16px;
    }

    .step-num {
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 600;
        flex-shrink: 0;
    }

    .step-text {
        color: rgba(255,255,255,0.6);
        font-size: 14px;
        line-height: 1.6;
        padding-top: 4px;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 16px;
    }

    .block-container {
        padding: 0 40px !important;
        max-width: 900px !important;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies["genres"] = movies["genres"].fillna("")
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
    genre = movies.loc[idx, "genres"]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    movie_indices = [s[0] for s in sim_scores]
    scores = [round(s[1], 3) for s in sim_scores]
    result = movies.loc[movie_indices].reset_index(drop=True)
    return result, scores, matched_title

movies = load_data()
cosine_sim = build_model(movies)

st.markdown("""
<div class="hero">
    <div class="hero-badge">🎬 AI Powered</div>
    <h1 class="hero-title">CineMatch</h1>
    <p class="hero-subtitle">Discover your next favourite movie using machine learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-number">100+</div>
        <div class="stat-label">Movies</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">TF-IDF</div>
        <div class="stat-label">Algorithm</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">Cosine</div>
        <div class="stat-label">Similarity</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">Top 5</div>
        <div class="stat-label">Recommendations</div>
    </div>
</div>
""", unsafe_allow_html=True)

movie_input = st.text_input(
    "🔍  Search a movie you love",
    placeholder="Try: Avatar, Matrix, Inception, Titanic..."
)

if not movie_input:
    st.markdown("""
    <div class="how-it-works">
        <div class="how-title">⚙️ How It Works</div>
        <div class="step-item">
            <div class="step-num">1</div>
            <div class="step-text"><strong style="color:white">Input</strong> — Type any movie name you enjoy in the search box above.</div>
        </div>
        <div class="step-item">
            <div class="step-num">2</div>
            <div class="step-text"><strong style="color:white">TF-IDF Vectorization</strong> — The system converts movie genres into numerical vectors using Term Frequency-Inverse Document Frequency.</div>
        </div>
        <div class="step-item">
            <div class="step-num">3</div>
            <div class="step-text"><strong style="color:white">Cosine Similarity</strong> — It calculates the angle between vectors to find movies closest in genre space.</div>
        </div>
        <div class="step-item">
            <div class="step-num">4</div>
            <div class="step-text"><strong style="color:white">Results</strong> — Top 5 most similar movies are returned ranked by similarity score.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if movie_input:
    result = get_recommendations(movie_input, movies, cosine_sim)
    if result[0] is None:
        st.markdown(f"""
        <div class="error-box">
            ❌ No results found for "<strong>{movie_input}</strong>"<br>
            <small style="opacity:0.7">Try: Avatar, Matrix, Inception, Titanic, Forrest Gump</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        recs, scores, matched = result
        st.markdown(f"""
        <div class="matched-banner">
            ✅ Showing recommendations based on: <span>{matched}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🍿 Top 5 Recommendations</div>', unsafe_allow_html=True)

        for i, (_, row) in enumerate(recs.iterrows(), 1):
            score = scores[i-1]
            bar_width = int(score * 100)
            genres_html = "".join([f'<span class="genre-tag">{g.strip()}</span>' for g in row["genres"].split() if g.strip()])
            st.markdown(f"""
            <div class="movie-card">
                <div class="card-rank">#{i} Recommendation</div>
                <div class="card-title">{row["title"]}</div>
                <div style="margin-bottom:14px">{genres_html}</div>
                <div class="score-row">
                    <div class="score-label">Match Score</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                    <div class="score-value">{score}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built with <span>Python · Scikit-learn · Streamlit</span> &nbsp;|&nbsp; Content-Based Filtering using TF-IDF & Cosine Similarity<br>
    <span>CineMatch</span> — Final Year AI/ML Project
</div>
""", unsafe_allow_html=True)