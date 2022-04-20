from argparse import ArgumentParser


def add_arguments(parser: ArgumentParser):
    parser.add_argument('--path_to_table', type=str,
                        help='Path to table file with recognized text',
                        required=True)
    parser.add_argument('--file_type', type=str,
                        help='To specify type (extension) of input table file (output data is always in csv format)',
                        default=None)
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
    parser.add_argument('--train_validation_test_split',
                        action='store_true',
                        help='Flag of dataset splitting on train, validation and test part')
    return parser
