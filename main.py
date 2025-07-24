import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge on movie title
movies = movies.merge(credits, on='title')

# Select relevant features
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Preprocessing functions
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L[:3]  # Top 3 only

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

def collapse(lst):
    return [i.replace(" ", "") for i in lst]

# Apply transformations
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['crew'] = movies['crew'].apply(get_director)
movies['crew'] = movies['crew'].apply(lambda x: x.replace(" ", ""))

# Combine into a single "tags" column
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast']
movies['tags'] = movies['tags'] + movies['crew'].apply(lambda x: [x])

# Final data
new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Vectorize text data
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return "Movie not found in dataset!"

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\nTop 5 recommendations for '{new_df.iloc[index].title}':\n")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Example usage
if __name__ == "__main__":
    recommend('Inception')
