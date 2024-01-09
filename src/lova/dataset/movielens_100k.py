import os
from typing import List

import pandas as pd

from .base import BaseDownloader


class MovieLens100KDownLoader(BaseDownloader):
    """
    A downloader class for the MovieLens 100K dataset. This class provides methods to download
    and read different parts of the MovieLens 100K dataset, such as ratings, users, genres, and items.

    Attributes:
        DOWNLOAD_URL (str): URL for downloading the MovieLens 100K dataset.
        DEFAULT_PATH (str): Default path to save the downloaded dataset.

    Methods:
        read_ratings: Reads the ratings from the dataset.
        read_users: Reads the user information from the dataset.
        _read_genres: Reads the genres from the dataset.
        read_items: Reads the movie items from the dataset.
    """

    DOWNLOAD_URL: str = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DEFAULT_PATH: str = os.path.join(os.getcwd(), "ml-100k")

    def read_ratings(self) -> pd.DataFrame:
        """
        Reads the ratings data from the dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the ratings data with columns ['userId', 'movieId', 'rating', 'timestamp'].
        """
        ratings_path: str = os.path.join(self.DEFAULT_PATH, "u.data")
        df_ratings: pd.DataFrame = pd.read_csv(
            ratings_path, sep="\t", header=None, engine="python"
        ).copy()
        df_ratings.columns = pd.Index(["userId", "movieId", "rating", "timestamp"])
        df_ratings["timestamp"] = pd.to_datetime(df_ratings.timestamp, unit="s")

        return df_ratings

    def read_users(self) -> pd.DataFrame:
        """
        Reads the user data from the dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the user data with columns ['userId', 'gender', 'age', 'occupation', 'zipcode'].
        """
        users_path: str = os.path.join(self.DEFAULT_PATH, "u.user")
        df_users: pd.DataFrame = pd.read_csv(
            users_path,
            sep="|",
            header=None,
            engine="python",
            names=["userId", "age", "gender", "occupation", "zipcode"],
        )

        return df_users[["userId", "gender", "age", "occupation", "zipcode"]]

    def _read_genres(self) -> List[str]:
        """
        Reads the genre data from the dataset.

        Returns:
            List[str]: A list of genres available in the dataset.
        """
        genres_path: str = os.path.join(self.DEFAULT_PATH, "u.genre")
        with open(genres_path, "r") as outfile:
            genres = outfile.read()
            return [pair.split("|")[0] for pair in genres.split("\n")][:-2]

    def read_items(self) -> pd.DataFrame:
        """
        Reads the movie items data from the dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the movie items data with columns ['itemId', 'title', 'genres', 'release_date'].
        """
        movies_path: str = os.path.join(self.DEFAULT_PATH, "u.item")
        genres: List[str] = self._read_genres()
        df_items: pd.DataFrame = pd.read_csv(
            movies_path,
            sep="|",
            header=None,
            encoding="latin-1",
            engine="python",
            names=["itemId", "title", "release_date", "video_release_date", "URL"]
            + genres,
        )
        df_items["title"] = df_items["title"].str[:-7]
        df_items["title"] = df_items["title"].apply(lambda x: x.split(",")[0])
        df_items["release_date"] = pd.to_datetime(df_items.release_date)
        df_items["genres"] = df_items[genres] @ (df_items[genres].columns + "|")
        df_items["genres"] = df_items["genres"].apply(lambda x: x[:-1].split("|"))
        df_items = df_items.drop(columns=genres + ["video_release_date", "URL"])
        return df_items[["itemId", "title", "genres", "release_date"]]
