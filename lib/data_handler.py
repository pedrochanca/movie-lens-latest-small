import pandas as pd
import numpy as np
from typing import Optional, Any, Dict


class MovieLens:

    def __init__(self):
    
        self.links, self.movies, self.ratings, self.tags = self._load_data()
        self._transform_tags()
        self._transform_movies()
        self.users = self._compute_users()

    def _load_data(self):

        file_directory = "/workspaces/movie-lens-latest-small/data"

        links = load_csv(f"{file_directory}/links.csv")
        movies = load_csv(f"{file_directory}/movies.csv")
        ratings = load_csv(f"{file_directory}/ratings.csv")
        tags = load_csv(f"{file_directory}/tags.csv")

        return links, movies, ratings, tags

    def _transform_tags(self):

        self.tags = (
            self.tags
            .groupby(["userId", "movieId"])
            .agg(lambda x: ', '.join(map(str, x))).reset_index()
        )

    def _transform_movies(self):

        # Extract title and year using a regex and create new columns
        self.movies['release_year'] = (
            self.movies.loc[:, 'title'].str.extract(r'\((\d{4})\)')
        )
        self.movies['title'] = (
            self.movies.loc[:, 'title'].str.extract(r'^(.*) \(\d{4}\)')
        )
        self.movies['genres'] = self.movies['genres'].str.replace('|', ', ', regex=False)

        self.movies = pd.merge(self.movies, self.links, on=["movieId"], how="left")

    def _compute_users(self):
        
        return pd.merge(
            self.ratings, self.tags, 
            on=["userId", "movieId"], how="left", suffixes=('_ratings', '_tags')
        )


def load_csv(file_path: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    """
    Load CSV data from a given file path.

    Parameters:
    - file_path (str): The path to the CSV file.
    - **kwargs: Additional arguments to pass to pandas.read_csv.

    Returns:
    - DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        print(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":

    movie_lens = MovieLens()

    print(movie_lens.movies)
    print(movie_lens.users)


