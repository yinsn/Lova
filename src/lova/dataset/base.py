import os
import zipfile
from abc import ABCMeta
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve


class BaseDownloader(metaclass=ABCMeta):
    """Base downloader for all Movielens datasets.

    This class provides a framework for downloading and extracting the Movielens datasets.
    It ensures that the dataset is downloaded from a specified URL and extracted to a given path.

    Attributes:
        DOWNLOAD_URL (str): The URL from where the dataset will be downloaded.
        DEFAULT_PATH (str): The default path where the dataset will be stored.
        zip_path (Path): The path where the dataset zip file will be stored.

    Args:
        zip_path (Optional[Union[Path, str]]): The path where the dataset zip file should be stored.
            If None, the DEFAULT_PATH is used.
    """

    DOWNLOAD_URL: str
    DEFAULT_PATH: str

    def __init__(self, zip_path: Optional[Union[Path, str]] = None) -> None:
        """Initializes the BaseDownloader object with the provided zip path or the default path."""
        if zip_path is None:
            zip_path = self.DEFAULT_PATH
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            self._retrieve()

    def _retrieve(self) -> None:
        """Downloads and extracts the dataset.

        Retrieves the dataset from the DOWNLOAD_URL and extracts it to the specified path.
        The original zip file is removed after extraction.
        """
        url: str = self.DOWNLOAD_URL
        file_name: str = str(self.zip_path) + ".zip"
        urlretrieve(url, filename=file_name)
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall(self.zip_path.parent)
        os.remove(file_name)
