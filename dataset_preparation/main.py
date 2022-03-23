# -*- coding: utf-8 -*-
import os
import argparse
import re
import pandas as pd

from utils import parser


def get_relative_path_gen(images_dir_path: str, csv_path: str):
    root = os.path.commonpath([images_dir_path, csv_path])
    relative_path_to_image_dir = images_dir_path[len(root) + 1:]

    def get_relative_image_path(img_path: str):
        return relative_path_to_image_dir + "/" + os.path.basename(img_path)

    return get_relative_image_path


def csv_format(csv_path: str, image_path_column: int, text_column: int, images_dir: str):
    csv_data = pd.read_csv(csv_path)
    # Renaming two columns with path to images and recognized text
    columns = csv_data.columns.values
    csv_data = csv_data.rename(columns={
        columns[image_path_column]: 'filename',
        columns[text_column]: 'text'})
    csv_data = csv_data[['filename', 'text']]
    # Deleting empty rows
    csv_data = csv_data.dropna()
    # Check path to images
    get_relative_image_path = get_relative_path_gen(images_dir, csv_path)
    csv_data['filename'] = csv_data['filename'].apply(get_relative_image_path)
    # Saving fixed dataset
    csv_data.to_csv(csv_path, index=False, encoding="utf-8")


def csv_generation(images_dir: str, text_dir: str, csv_path: str):
    # Getting list of image files
    image_extensions = [".jpg", ".png"]
    image_list = list(filter(lambda file: re.match(r".*(" + "|".join(image_extensions)+")", file),
                             os.listdir(images_dir)))

    # Matching images with texts
    filename = []
    text = []
    for image_file in image_list:
        image_name = os.path.splitext(image_file)[0]
        text_file = text_dir + "/" + image_name + ".txt"
        try:
            with open(text_file, 'r', encoding="utf-8") as f:
                filename.append(image_file)
                text.append(f.read())
        except FileNotFoundError:
            pass

    # Saving information in csv table
    csv_data = pd.DataFrame({"filename": filename, "text": text})
    csv_data.to_csv(csv_path, index=False, encoding="utf-8")


if __name__ == '__main__':
    parser = parser.add_arguments(argparse.ArgumentParser())
    args = parser.parse_args()

    if args.texts_dir is not None:
        csv_generation(args.images_dir, args.texts_dir, args.csv_path)

    csv_format(args.csv_path, args.image_path_column, args.text_column,
               args.images_dir)

    print("INFO: Success!")
