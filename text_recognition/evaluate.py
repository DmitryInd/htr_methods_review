import torch
import argparse

from utils.dataset import get_data_loader
from utils.assessment import val_loop
from utils.transforms import get_val_transforms
from utils.config import Config
from utils.tokenizer import get_tokenizer
from models.builder import get_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config):
    tokenizer = get_tokenizer(config.get('tokenizer'), config)

    val_transforms = get_val_transforms(
        height=config.get_image('height'),
        width=config.get_image('width')
    )
    test_loader = get_data_loader(
        transforms=val_transforms,
        csv_paths=config.get_test_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=config.get_test_datasets('prob'),
        epoch_size=config.get_test('epoch_size'),
        batch_size=config.get_test('batch_size'),
        drop_last=False
    )

    model = get_model(config.get("model"), number_class_symbols=tokenizer.get_num_chars(), config=config)
    model.load_state_dict(torch.load(args.model_path))
    model.to(DEVICE)

    val_loop(test_loader, model, tokenizer, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='configs/ocr_config.json',
                        help='Path to config.json.')
    parser.add_argument('--model_path', type=str,
                        help='Path to model weights.')
    args = parser.parse_args()

    main(config=Config(args.config_path))
