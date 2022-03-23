from argparse import ArgumentParser


def add_arguments(parser: ArgumentParser):
    parser.add_argument('--csv_path', type=str,
                        help='Path to csv file with recognized text',
                        required=True)
    parser.add_argument('--image_path_column', type=int,
                        default=0,
                        help='Number of column with paths to images')
    parser.add_argument('--text_column', type=int,
                        default=1,
                        help='Number of column with recognized text')
    parser.add_argument('--images_dir', type=str,
                        help='Path to folder with images',
                        required=True)
    parser.add_argument('--texts_dir', type=str,
                        help='Path to folder with recognized text')
    parser.add_argument('--train_validation_test_split', type=bool,
                        default=False,
                        help='Flag of dataset splitting on train, validation and test part')
    return parser
