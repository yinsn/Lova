import logging
import os
import re
from typing import Dict, Optional

import pandas as pd

from .base import BaseDataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SequenceLoader(BaseDataLoader):
    def __init__(
        self,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_type: str = "csv",
        max_rows: Optional[int] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the SequenceLoader object.

        Args:
            file_path (str, optional): The path to the directory where the file is located.
            file_name (str, optional): The name of the file without extension.
            file_type (str, optional): The type of the file. Defaults to 'csv'.
            max_rows (Optional[int], optional): Maximum number of rows to load from the file. Defaults to None.
            config (Optional[Dict], optional): A dictionary containing the configuration for the data loader. Defaults to None.
        """
        super().__init__(file_path, file_name, file_type, max_rows, config)
        self.file_url = (
            os.path.join(self.file_path, self.file_name) + "." + self.file_type
        )

    def _get_column_names(self) -> None:
        """
        Private method to extract column names from the first line of the file.
        """
        with open(self.file_url) as f:
            headline = f.readline().strip("\n")
            self.column_names = [item.split(".")[1] for item in headline.split(",")]

    @staticmethod
    def _extract_sequence(string: str) -> list:
        """
        Static method to extract a sequence of addresses from a given string.

        Args:
            string (str): The string containing the sequence.

        Returns:
            List[str]: A list of extracted addresses.

        Note: Set the first two columns as ID information by default, excluding them from sequence features
        """
        first_split = string.strip("\n").split("\x00")
        if len(first_split) > 1:
            split_addresses = first_split[0].split(",") + first_split[1:]
            split_addresses = [i for i in split_addresses if i != "" and i != ","]
        else:
            split_addresses = first_split[0].split(",")
        return split_addresses

    @staticmethod
    def convert_string_to_tuple_of_num(string: str) -> tuple:
        """
        Converts a string containing numeric values to a tuple of numbers.

        This method uses regular expressions to find all integer and floating-point numbers in the input string.
        Each number is converted to an integer if it's a whole number, or to a float if it has a decimal point.

        Args:
            string (str): The string containing numeric values.

        Returns:
            Tuple[Union[int, float], ...]: A tuple containing the extracted numbers, each as an integer or a float.

        Example:
            convert_string_to_tuple_of_num("12, -34.5") returns (12, -34.5)
        """
        numbers = re.findall(r"[-+]?\d+\.?\d*", string)
        num_tuple = tuple(
            int(num) if num.replace("-", "").isdigit() else float(num)
            for num in numbers
        )
        return num_tuple

    def _get_dataframe(self) -> None:
        """
        Private method to load data from the file and convert it into a pandas DataFrame.
        """
        logger.info("Loading data from %s", self.file_url)
        split_block = []
        with open(self.file_url) as f:
            lines = f.readlines()[1:]
            if self.max_rows is not None:
                logger.info("Only loading %s rows", self.max_rows)
                lines = lines[: self.max_rows]
            for line in lines:
                split_block.append(self._extract_sequence(line))
        logger.info("Loading data finished")
        self.dataframe = pd.DataFrame(split_block, columns=self.column_names)

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
        Load data from the file and return it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        self._get_column_names()
        self._get_dataframe()
        self._literal_dataframe()
        logger.info("Data loading finished")
        return self.dataframe
