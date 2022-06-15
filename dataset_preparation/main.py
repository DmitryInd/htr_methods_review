# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import re

from dataset_tools import parser, file_handler


def get_sequence_statistics(csv_path: str):
    csv_data = pd.read_csv(csv_path, encoding="utf-8")
    text_lens = csv_data['text'].apply(lambda sequence: len(sequence))
    return text_lens.max(), text_lens.mean()


def get_alphabet(csv_path: str, base_alphabet=None):
    alphabet = set()
    if base_alphabet is not None:
        alphabet.update(base_alphabet)

    csv_data = pd.read_csv(csv_path, encoding="utf-8")
    for text in csv_data['text'].to_list():
        alphabet.update(text)
    return sorted(list(alphabet))


def train_validation_test_split(csv_path: str, train_part=.8, validation_part=.1):
    csv_data = pd.read_csv(csv_path, encoding="utf-8")
    train, validate, test = np.split(csv_data.sample(frac=1, random_state=42),
                                     [int(train_part * len(csv_data)),
                                      int((train_part + validation_part) * len(csv_data))])

    csv_path = file_handler.get_pure_file_name(csv_path)
    train.to_csv(csv_path + "_train.csv", index=False, encoding="utf-8")
    validate.to_csv(csv_path + "_validation.csv", index=False, encoding="utf-8")
    test.to_csv(csv_path + "_test.csv", index=False, encoding="utf-8")


def get_relative_path_gen(images_dir_path: str, csv_path: str):
    root = os.path.commonpath([images_dir_path, csv_path])
    relative_path_to_image_dir = images_dir_path[len(root) + 1:]

    def get_relative_image_path(img_path: str):
        return relative_path_to_image_dir + "/" + os.path.basename(img_path)

    return get_relative_image_path


def csv_format(path_to_file: str, file_extension: str, image_path_column: int, text_column: int, images_dir: str, dataset_len: int):
    file_reader = file_handler.get_file_reader(file_extension)
    csv_data = file_reader(path_to_file)
    # Renaming two columns with path to images and recognized text
    columns = csv_data.columns.values
    csv_data = csv_data.rename(columns={
        columns[image_path_column]: 'filename',
        columns[text_column]: 'text'})
    csv_data = csv_data[['filename', 'text']]
    # Deleting empty rows
    csv_data = csv_data.dropna()
    # Generating relative path to images
    get_relative_image_path = get_relative_path_gen(images_dir, path_to_file)
    csv_data['filename'] = csv_data['filename'].apply(get_relative_image_path)
    # Limit size of dataset
    if dataset_len > 0:
        csv_data = csv_data.sample(n=dataset_len, random_state=42)
    # Saving fixed dataset in csv format
    path_to_file = file_handler.get_pure_file_name(path_to_file) + ".csv"
    csv_data.to_csv(path_to_file, index=False, encoding="utf-8")
    return path_to_file


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
        csv_generation(args.images_dir, args.texts_dir, args.path_to_table)

    file_type = args.file_type if args.file_type is not None else file_handler.get_file_type(args.path_to_table)
    args.path_to_table = csv_format(args.path_to_table, file_type,
                                    args.image_path_column, args.text_column,
                                    args.images_dir, args.dataset_len)

    if args.train_validation_test_split:
        train_validation_test_split(args.path_to_table)

    print("Symbols in dataset:\n" + ''.join(get_alphabet(args.path_to_table)))
    max_sequence_len, mean_sequence_len = get_sequence_statistics(args.path_to_table)
    print(f"Max len of sequence: {max_sequence_len}, average len of sequence: {mean_sequence_len:.3f}")

    print("INFO: Success!")
