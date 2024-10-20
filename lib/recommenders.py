import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:

    def __init__(
        self, movies: pd.DataFrame, features_columns: list, verbose: bool=False
    ):

        self.movies = movies
        self.features_columns = features_columns
        self.verbose = verbose

        self._add_features_column()

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['features'])

        if verbose:
            print(self.tfidf_matrix)
            print(self.get_feature_names())

    def get_recommendations_by_features(self, features: list, n=5):
        # compute tfidf for each user feature, using the pre-trainined tfidf
        input_vector = self.tfidf_vectorizer.transform([features])
        # calculate similarity with all movies
        sim_scores = self._get_similarity_scores(input_vector)

        # get top N similar movies
        top_indices = sim_scores.argsort()[::-1][:n]
        return self.movies.iloc[top_indices]
    

    def get_recommendations_by_title(self, title: str, n=5):
        title_index = self.movies[self.movies['title'] == title].index[0]
        input_vector = self.tfidf_matrix[title_index]

        sim_scores = self._get_similarity_scores(input_vector)

        top_indices = sim_scores.argsort()[::-1]
        top_indices = [idx for idx in top_indices if idx != title_index][:n]

        return self.movies.iloc[top_indices]

    
    def get_feature_names(self):
        return self.tfidf_vectorizer.get_feature_names_out()
    
    def _add_features_column(self):
        
        features = ''

        for i, feature in enumerate(self.features_columns):
            if i != 0:
                features += ', '

            features += self.movies[feature]

        self.movies["features"] = features

    def _get_similarity_scores(self, input_vector, measure: str="cosine"):
        if measure == 'cosine':
            return cosine_similarity(input_vector, self.tfidf_matrix).flatten()




if __name__ == "__main__":
    
    # Example usage
    # -------------
    data = {
        "title": [
            "The Matrix", "Avengers: Endgame", "Inception", "Titanic", "Forrest Gump"
        ],
        "genre": [
            "Action, Sci-fi", "Action, Adventure, Sci-fi", "Sci-fi, Thriller", 
            "Romance, Drama", "Comedy, Drama"
        ],
        "keywords": [
            "Time Travel, Matrix, Dystopia", "Superhero, Infinity War", "Dream, Illusion", 
            "Love, Shipwreck", "Friendship, Life"
        ]
    }

    movies_df = pd.DataFrame(data)

    # Initialize the recommender
    recommender = ContentBasedRecommender(
        movies_df, ["genre", "keywords"], verbose=False
    )

    # Get movie recommendation based in user features
    features = "sci-fi drama"
    recommendations = recommender.get_recommendations_by_features(features, 2)
    print(features)
    print(recommendations["title"].values)

    # Get movie recommendation based in user watched titles
    title = "The Matrix"
    recommendations = recommender.get_recommendations_by_title(title, 2)
    print(title)
    print(recommendations["title"].values)