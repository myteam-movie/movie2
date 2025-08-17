import streamlit as st
import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None

def convert_genres_keywords(obj):
    """Convert genres and keywords from string to list"""
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

def convert_cast(obj):
    """Convert cast from string to list (top 3 actors)"""
    try:
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L
    except:
        return []

def fetch_director(obj):
    """Extract director from crew"""
    try:
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    except:
        return []

def stem_text(text):
    """Apply stemming to text"""
    ps = PorterStemmer()
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the movie data from local files"""
    try:
        # Try to load the data files
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        
        # Check if credits.csv.gz exists, otherwise try credits.csv
        if os.path.exists('credits.csv.gz'):
            credits_df = pd.read_csv('credits.csv.gz', compression='gzip')
        elif os.path.exists('tmdb_5000_credits.csv'):
            credits_df = pd.read_csv('tmdb_5000_credits.csv')
        else:
            st.error("Credits file not found. Please ensure 'credits.csv.gz' or 'tmdb_5000_credits.csv' exists.")
            return None, None
        
        # Merge datasets
        movies = movies_df.merge(credits_df, on='title')
        
        # Select required columns
        movies = movies[['movie_id', 'cast', 'crew', 'title', 'overview', 'genres', 'keywords']]
        
        # Drop null values
        movies.dropna(inplace=True)
        
        # Convert string representations to lists
        movies['genres'] = movies['genres'].apply(convert_genres_keywords)
        movies['keywords'] = movies['keywords'].apply(convert_genres_keywords)
        movies['cast'] = movies['cast'].apply(convert_cast)
        movies['crew'] = movies['crew'].apply(fetch_director)
        
        # Convert overview to list of words
        movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
        
        # Remove spaces from names
        movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
        
        # Create tags by combining all features
        movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        
        # Create new dataframe with required columns
        new_df = movies[['movie_id', 'title', 'tags']].copy()
        
        # Convert tags to string
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
        
        # Apply stemming
        new_df['tags'] = new_df['tags'].apply(stem_text)
        
        return new_df, True
        
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, False
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None, False

@st.cache_data
def create_similarity_matrix(df):
    """Create similarity matrix using CountVectorizer and cosine similarity"""
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vector)
    return similarity

def recommend_movies(movie_title, df, similarity_matrix, num_recommendations=5):
    """Recommend movies based on similarity"""
    try:
        movie_index = df[df['title'] == movie_title].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(df.iloc[i[0]]['title'])
        
        return recommended_movies
    except IndexError:
        return None

# Streamlit App
def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")
    
    # Load and process data automatically
    with st.spinner("Loading and processing movie data..."):
        movies_df, success = load_and_preprocess_data()
    
    if not success or movies_df is None:
        st.error("""
        ❌ **Error loading data files!**
        
        Please ensure you have the following files in your repository:
        - `tmdb_5000_movies.csv`
        - `credits.csv.gz` OR `tmdb_5000_credits.csv`
        
        The app cannot proceed without these data files.
        """)
        return
    
    # Create similarity matrix
    with st.spinner("Building recommendation engine..."):
        similarity_matrix = create_similarity_matrix(movies_df)
    
    st.success(f"✅ Loaded {len(movies_df)} movies successfully!")
    
    # Main recommendation interface
    st.subheader("🔍 Get Movie Recommendations")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Movie selection
        movie_titles = sorted(movies_df['title'].tolist())
        selected_movie = st.selectbox(
            "Choose a movie you like:",
            options=movie_titles,
            index=None,
            placeholder="Start typing to search for a movie..."
        )
    
    with col2:
        # Number of recommendations
        num_recs = st.slider("Number of recommendations:", 1, 10, 5)
    
    if selected_movie:
        if st.button("🎯 Get Recommendations", type="primary"):
            recommendations = recommend_movies(
                selected_movie, 
                movies_df, 
                similarity_matrix, 
                num_recs
            )
            
            if recommendations:
                st.subheader(f"🍿 Movies similar to '{selected_movie}':")
                
                # Display recommendations in a nice format
                for i, movie in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {movie}")
                
                # Show some statistics
                st.info(f"Found {len(recommendations)} recommendations based on genres, cast, crew, and plot similarities.")
                
            else:
                st.error("❌ Movie not found or no recommendations available. Please try a different movie.")
    
    # Additional information
    with st.expander("📊 Dataset Information"):
        st.write(f"**Total movies in dataset:** {len(movies_df):,}")
        st.write("**Sample movies:**")
        
        # Show random sample of movies
        sample_movies = movies_df['title'].sample(min(10, len(movies_df))).tolist()
        for movie in sample_movies:
            st.write(f"• {movie}")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This recommendation system uses **content-based filtering** to suggest similar movies:
        
        1. **Feature Extraction**: Combines movie genres, keywords, cast, crew, and plot overview
        2. **Text Processing**: Applies stemming and removes stop words
        3. **Vectorization**: Converts text features into numerical vectors using CountVectorizer
        4. **Similarity Calculation**: Uses cosine similarity to find similar movies
        5. **Recommendations**: Returns top N most similar movies
        
        The system focuses on content similarity rather than user ratings or collaborative filtering.
        """)

if __name__ == "__main__":
    main()
