import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import requests

# -------------------- Data Loading --------------------
@st.cache_data
def load_data():
    credits = pd.read_csv("credits.csv.gz", compression="gzip")
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id', 'cast', 'crew', 'title', 'overview', 'genres', 'keywords']]
    movies.dropna(inplace=True)

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id','title','tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(i) for i in text.split()])

    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vector)

    return new_df, similarity

new_df, similarity = load_data()

# -------------------- Recommendation --------------------
def recommend(movie):
    try:
        movie_index = new_df[new_df['title'].str.lower() == movie.lower()].index[0]
    except IndexError:
        return []
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]

# -------------------- Streamlit UI --------------------
st.title("🎬 Movie Recommender System")
st.markdown("Pick a movie and get 5 similar movies recommended to you!")

# Dropdown menu
selected_movie = st.selectbox("🎥 Choose a movie", new_df['title'].values)

# Button
if st.button("Recommend"):
    recs = recommend(selected_movie)
    if not recs:
        st.error("❌ No movies found. Try another title.")
    else:
        st.subheader("✨ Recommended Movies")
        for r in recs:
            st.success(r)
