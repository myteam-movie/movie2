import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# -----------------------
# Data preprocessing
# -----------------------

@st.cache_data
import streamlit as st

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Example UI
st.markdown("<div class='title'>🎬 Movie Recommender System</div>", unsafe_allow_html=True)
movies = ["Inception", "Interstellar", "The Dark Knight"]
for m in movies:
    st.markdown(f"<div class='recommend-box'>{m}</div>", unsafe_allow_html=True)

def load_data():
    # Load data (make sure the files are in the same folder or provide paths)
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv("credits.csv.gz", compression="gzip")

    # Merge
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'cast', 'crew', 'title', 'overview', 'genres', 'keywords']]
    movies.dropna(inplace=True)

    # Parse JSON-like columns
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert3(obj):
        L = []
        Counter = 0
        for i in ast.literal_eval(obj):
            if Counter != 3:
                L.append(i['name'])
                Counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)

    # Clean text
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    # Stemming
    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(i) for i in text.split()])

    new_df['tags'] = new_df['tags'].apply(stem)

    # Vectorize
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vector)

    return new_df, similarity

movies_df, similarity = load_data()

# -----------------------
# Recommendation function
# -----------------------

def recommend(movie):
    if movie not in movies_df['title'].values:
        return []
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies_df.iloc[i[0]].title for i in movies_list]

# -----------------------
# Streamlit UI
# -----------------------

st.title("🎬 Movie Recommender System")
st.write("Select a movie and get 5 similar recommendations.")

selected_movie = st.selectbox("Choose a movie:", movies_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader("Top 5 Recommendations:")
        for i, rec in enumerate(recommendations, start=1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("Movie not found in database.")
