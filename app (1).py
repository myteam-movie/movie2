import streamlit as st

# Movie dataset (example: only first few shown, extend to 50 movies)
movies = [
    {"id":1,"title":"Batman Begins","genres":["Action","Crime","Drama"],"year":2005,"rating":8.2,"poster":"https://image.tmdb.org/t/p/w500/8RW2runSEc34IwKN2D1aPcJd3P3.jpg","overview":"After witnessing his parents’ murder, Bruce Wayne trains with the League of Shadows before returning to Gotham to become Batman and fight corruption."},
    {"id":2,"title":"The Dark Knight","genres":["Action","Crime","Drama"],"year":2008,"rating":9.0,"poster":"https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg","overview":"Batman faces his greatest psychological and physical tests when the Joker unleashes chaos upon Gotham City."},
    {"id":3,"title":"Inception","genres":["Action","Sci-Fi","Thriller"],"year":2010,"rating":8.8,"poster":"https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg","overview":"A skilled thief who steals corporate secrets through dream-sharing technology is given a chance to have his past crimes forgiven if he implants an idea into someone’s subconscious."},
    {"id":4,"title":"Interstellar","genres":["Adventure","Drama","Sci-Fi"],"year":2014,"rating":8.6,"poster":"https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg","overview":"A team of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival as Earth becomes uninhabitable."},
    {"id":5,"title":"The Matrix","genres":["Action","Sci-Fi"],"year":1999,"rating":8.7,"poster":"https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg","overview":"A computer hacker learns about the true nature of reality and his role in the war against the controllers of it."}
    # ... continue with rest of 50 movies
]

# Streamlit UI
st.title("🎬 MovieFinder")
st.markdown("Discover your next favorite movie with smart recommendations")

# Search bar
search_term = st.text_input("🔎 Search for a movie").lower()

# Filter movies
results = [m for m in movies if search_term in m["title"].lower()] if search_term else movies

# Display results
if results:
    for m in results:
        st.image(m["poster"], width=250)
        st.subheader(f"{m['title']} ({m['year']})")
        st.markdown(f"⭐ **{m['rating']}**  |  🎭 {', '.join(m['genres'])}")
        st.markdown(f"_{m['overview']}_")
        st.markdown("---")
else:
    st.warning("No movies found")
