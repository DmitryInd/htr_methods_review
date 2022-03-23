import os
import argparse
import numpy as np
import pandas as pd

from utils import parser


def get_relative_path_gen(images_path: str, csv_path: str):
    root = os.path.commonpath([images_path, csv_path])
    relative_path_to_image_dir = images_path[len(root) + 1:]

    def get_relative_image_path(img_path: str):
        return relative_path_to_image_dir + "/" + os.path.basename(img_path)

    return get_relative_image_path


def csv_preparation(csv_path: str, image_path_column: int, text_column: int, images_path: str):
    csv_data = pd.read_csv(csv_path)
    # Renaming first tow columns with path to images and recognized text
    columns = csv_data.columns.values
    csv_data = csv_data.rename(columns={
        columns[image_path_column]: 'filename',
        columns[text_column]: 'text'})
    csv_data = csv_data[['filename', 'text']]
    # Deleting empty rows
    csv_data = csv_data.dropna()
    # Check path to images
    get_relative_image_path = get_relative_path_gen(images_path, csv_path)
    csv_data['filename'] = csv_data['filename'].apply(get_relative_image_path)
    # Saving fixed dataset
    csv_data.to_csv(csv_path, index=False)


if __name__ == '__main__':
    parser = parser.add_arguments(argparse.ArgumentParser())
    args = parser.parse_args()
    if args.csv_path is not None:
        csv_preparation(args.csv_path, args.image_path_column, args.text_column,
                        args.images_dir)

    print("INFO: Success!")
