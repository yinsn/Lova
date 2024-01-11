from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd


class BaseDataLoader(metaclass=ABCMeta):
    """
    Initialize the BaseDataLoader object.

    This abstract base class defines the structure for data loaders. Implementations of this class should
    override the `load_data` method to load data from a specified source.

    Args:
        file_path (str): The path to the directory where the file is located.
        file_name (str): The name of the file without the extension.
        file_type (str, optional): The type of the file (e.g., 'csv'). Defaults to 'csv'.
        max_rows (Optional[int], optional): The maximum number of rows to load from the file. Defaults to None.

    """

    def __init__(
        self,
        file_path: str,
        file_name: str,
        file_type: str = "csv",
        max_rows: Optional[int] = None,
    ) -> None:
        self.file_path = file_path
        self.file_name = file_name
        self.file_type = file_type
        self.max_rows = max_rows

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Abstract method to load data into a pandas DataFrame.

        Subclasses must implement this method. It should read data from a source, process it as necessary,
        and return it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
