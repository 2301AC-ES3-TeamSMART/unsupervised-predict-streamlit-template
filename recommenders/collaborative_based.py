"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from functools import lru_cache

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

@lru_cache(maxsize=128)  # Add caching to the function
def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(ratings_df, reader)
    trainset, _ = train_test_split(data, test_size=0.01)  # Using a small test set for better performance

    svd_model = SVD(n_factors=200, n_epochs=40, lr_all=0.005, reg_all=0.02)
    svd_model.fit(trainset)

    predictions = []
    for ui in trainset.all_users():
        predictions.append(svd_model.predict(iid=item_id, uid=ui, verbose=False))

    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store = []
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def collab_model(movie_list, top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : int, optional
        Number of top recommendations to return to the user. Default is 10.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    indices = pd.Series(movies_df['title'])
    movie_ids = pred_movies(movie_list)
    df_init_users = ratings_df[ratings_df['userId'].isin(movie_ids)]
    # Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(df_init_users.pivot(index='userId', columns='movieId', values='rating').fillna(0))
    idxs = indices[indices.isin(movie_list)].index
    rank = cosine_sim[idxs]
    # Calculating the scores
    score_series = pd.Series(rank).sort_values(ascending=False)
    # Appending the names of movies
    listings = score_series.append(score_series).append(score_series).sort_values(ascending=False)
    recommended_movies = []
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes, idxs)
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])
    return recommended_movies

