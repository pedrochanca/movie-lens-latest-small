import pandas as pd
import numpy as np
import re

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

        def extract_title_and_year(title):
            # Match titles with year in parentheses at the end
            match = re.match(r'^(.*?)(?:\s*\((\d{4})\))?$', title)
            if match:
                extracted_title = match.group(1).strip()
                year = match.group(2)
                return extracted_title, year
            
            return title, None

        # Apply the function to create new columns
        self.movies['extracted_title'], self.movies['release_year'] = zip(
            *self.movies['title'].apply(extract_title_and_year)
        )
        # Update the original 'title' column
        self.movies['title'] = self.movies['extracted_title']
        # Drop the temporary 'extracted_title' column
        self.movies.drop('extracted_title', axis=1, inplace=True)

        # Replace NaN values in the 'release_year' column with an empty string
        self.movies['release_year'] = self.movies['release_year'].fillna('')

        self.movies['genres'] = (
            self.movies['genres'].str.replace('|', ', ', regex=False)
        )

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


