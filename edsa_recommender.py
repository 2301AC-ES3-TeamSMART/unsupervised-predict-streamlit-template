"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","SMARTai"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "SMARTai":
        st.title("About SMART Solutions")
        st.write("We are a team of smart, dedicated and motivated data scientist\
                    with the drive to profer updated solutions and modern \
                    approach to solving problems across different insdustries")
        st.title("What we have here is a recommendaion system, and this solution\
                    is implemented in Streaming Industry, E-commerce and lots\
                    more")
    #------------------------------------------------------------------------
    #--------------------Image Slider----------------------------------------
        st.title("Welcome to SMART Recommender")

        # Assuming you have a list of image paths or URLs
        image_paths = [
            "resources/imgs/slide/Image1.jpg",,
            "resources/imgs/slide/image2.jpg",
            "resources/imgs/slide/image3.jpg",
            "resources/imgs/slide/image4.jpg",
            "resources/imgs/slide/image5.jpg",
            "resources/imgs/slide/image6.jpg",
            "resources/imgs/slide/image7.jpg",
            "resources/imgs/slide/image8.jpg",
            "resources/imgs/slide/image9.jpg",
            "resources/imgs/slide/image10.jpg",
        ]

    # Display a slider widget to choose the index of the image
        selected_index = st.slider("Select an image", 0, len(image_paths) - 1, 0)

        # Display the selected image based on the slider value
        st.image(image_paths[selected_index], use_column_width=True)
    # You may want to add more sections here for aspects such as an EDA,

        st.title("Here, we will break down our approach and show you why this\
                    is important to you")

    # or to provide your business pitch.


if __name__ == '__main__':
    main()
