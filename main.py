import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data.csv')


# defining a function that calculates similarity matrix
def create_similarity_matrix():
    df = pd.read_csv('data.csv')
    # create count matrix from the above data table
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['comb'])
    # compute the cosine similarity based on the count matrix
    cosine_sim = cosine_similarity(count_matrix)
    return df,cosine_sim

# defining a function that recommends 10 most similar movies
def recommend(movie):
    movie = movie.lower()
    data, sim = create_similarity_matrix()
    # check if the movie is in our database or not
    if movie not in data['movie_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        movie_index = data.loc[data['movie_title']==movie].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        sim_movies = list(enumerate(sim[movie_index]))

        # sorting this list in decreasing order based on the similarity score
        sim_movies = sorted(sim_movies, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        sim_movies = sim_movies[1:11]

        # making an empty list that will containg all 10 movie recommendations
        recommend_movie = []
        for i in range(len(sim_movies)):
            a = sim_movies[i][0]
            recommend_movie.append(data['movie_title'][a])
        return recommend_movie

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = recommend(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run(debug=True)
