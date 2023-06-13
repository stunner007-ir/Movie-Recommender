from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv("raw_movies.csv")
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    data[feature] = data[feature].fillna('')

combined_features = data['genres'] + ' ' + data['keywords'] + ' ' + data['tagline'] + ' ' + data['cast'] + ' ' + data[
    'director']


def getRecommendations(movie_name):
    try:
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)
        similarity = cosine_similarity(feature_vectors)
        list_of_all_titles = data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        close_match = find_close_match[0]
        index_of_the_movie = data[data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        recommended_movies = []
        i = 1
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = data[data.index == index]['title'].values[0]
            if i <= 30:
                recommended_movies.append(title_from_index)
                i += 1
        return recommended_movies
    except:
        return "Sorry No suggestions available !"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    features = [str(x) for x in request.form.values()]
    print(features)
    movie_name = str(features[0])
    print(movie_name)
    output = getRecommendations(movie_name)
    return render_template('index.html', recommended_movie=output)


if __name__ == "__main__":
    app.run(debug=True)
