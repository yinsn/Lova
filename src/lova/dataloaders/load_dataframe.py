import logging
import os
from typing import Dict, Optional, Union

import pandas as pd

from .base import BaseDataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataFrameLoader(BaseDataLoader):
    def __init__(
        self,
        column_names: Optional[pd.Index] = None,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_type: str = "pkl",
        max_rows: Optional[int] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initializes the DataFrameLoader with specified parameters.

        Args:
            column_names (pd.Index): Index object representing the column names of the DataFrame.
            file_path (Optional[str]): Path of the file to be loaded. Defaults to None.
            file_name (Optional[str]): Name of the file to be loaded. Defaults to None.
            file_type (str): Type of the file to be loaded (e.g., 'csv', 'pkl'). Defaults to 'pkl'.
            max_rows (Optional[int]): Maximum number of rows to load from the file. Defaults to None.
            config (Optional[Dict]): Configuration parameters for data loading. Defaults to None.
        """
        super().__init__(file_path, file_name, file_type, max_rows, config)
        if config is not None:
            self.column_names = config.get("column_names", None)
        else:
            self.column_names = column_names

    @staticmethod
    def convert_string_to_tuple_of_num(value: Union[int, str]) -> tuple:
        """
        Convert a string or an integer into a tuple. If the input is a string, it
        is assumed to contain elements separated by '\x02'. Each element is either
        converted to an integer (if possible) or remains a string.

        Args:
            value (Union[int, str]): An integer or a string containing elements
                                    separated by '\x02'.

        Returns:
            Tuple[Union[int, str], ...]: A tuple where each element is either an
                                        integer or a string.
        """
        if isinstance(value, int):
            return (value,)
        else:
            elements = value.split("\x02")
            return tuple(
                int(element) if element.isdigit() else element for element in elements
            )

    def _get_dataframe(self) -> None:
        """
        Loads data from a file URL into a pandas DataFrame.
        """
        logger.info("Loading data from %s", self.file_path)
        if self.file_name is not None:
            file_url = os.path.join(self.file_path, self.file_name)
            data = pd.read_pickle(file_url)
        else:
            files = os.listdir(self.file_path)
            df_list = []
            for file in files:
                if file.endswith(self.file_type):
                    file_url = os.path.join(self.file_path, file)
                    data = pd.read_pickle(file_url)
                    df_list.append(data)
            data = pd.concat(df_list)

        if isinstance(data, pd.DataFrame):
            self.dataframe = data
            self.dataframe.columns = self.column_names
        else:
            logger.error("Failed to load data from %s", file_url)
            self.dataframe = pd.DataFrame()

    def _literal_dataframe(self) -> None:
        """
        Convert string representations in each column of the dataframe to their literal structures.
        """
        logger.info("Converting string representations to literal structures")
        for col in self.column_names:
            self.dataframe[col] = self.dataframe[col].apply(
                self.convert_string_to_tuple_of_num
            )

    def load_data(self) -> pd.DataFrame:
        """
        Loads data into a DataFrame, processes it, and returns the resulting DataFrame.

        Returns:
            pd.DataFrame: The loaded and processed DataFrame.
        """
        self._get_dataframe()
        self._literal_dataframe()
        logger.info("Data loading finished")
        return self.dataframe
