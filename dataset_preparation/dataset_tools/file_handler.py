import csv
import os
import pandas as pd
from abc import ABC, abstractmethod


def get_file_type(path_to_file: str):
    return os.path.splitext(path_to_file)[1][1:]


def get_pure_file_name(path_to_file: str):
    return os.path.splitext(path_to_file)[0]


def get_file_reader(file_type: str):
    if file_type == "csv":
        return CsvFileReader()
    elif file_type == "tsv":
        return TsvFileReader()

    return None


class FileReader(ABC):
    @abstractmethod
    def __call__(self, path_to_file: str) -> pd.DataFrame:
        pass


class CsvFileReader(FileReader):
    def __call__(self, path_to_file) -> pd.DataFrame:
        return pd.read_csv(path_to_file)


class TsvFileReader(FileReader):
    def __call__(self, path_to_file: str) -> pd.DataFrame:
        return pd.read_csv(path_to_file, sep="\t", quoting=csv.QUOTE_NONE)
