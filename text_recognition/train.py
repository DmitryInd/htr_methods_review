import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.assessment import val_loop, train_loop
from utils.weights_controller import FilesLimitControl, load_pretrain_model

from utils.dataset import get_data_loader
from utils.transforms import get_train_transforms, get_val_transforms
from utils.config import Config
from utils.tokenizer import get_tokenizer
from utils.criteria import get_criterion
from models.builder import get_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_plot(loss_history, train_cer_history, valid_cer_history):
    epoch_number = len(loss_history)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    ax[0].plot(loss_history, np.arange(epoch_number) + 1, label='train loss')
    ax[0].set_xlim(left=0)
    ax[0].set_xlabel('Epoch')
    ax[0].set_title('Train loss')
    ax[1].plot(train_cer_history, label='train cer history')
    ax[1].plot(valid_cer_history, label='valid cer history')
    ax[1].set_xlabel('Epoch')
    ax[1].set_title('CER')
    plt.legend()
    plt.grid()
    plt.show()


def get_loaders(tokenizer, config):
    train_transforms = get_train_transforms(
        height=config.get_image('height'),
        width=config.get_image('width'),
        prob=0.2
    )
    train_loader = get_data_loader(
        transforms=train_transforms,
        csv_paths=config.get_train_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=config.get_train_datasets('prob'),
        epoch_size=config.get_train('epoch_size'),
        batch_size=config.get_train('batch_size'),
        drop_last=True
    )
    val_transforms = get_val_transforms(
        height=config.get_image('height'),
        width=config.get_image('width')
    )
    val_loader = get_data_loader(
        transforms=val_transforms,
        csv_paths=config.get_val_datasets('csv_path'),
        tokenizer=tokenizer,
        dataset_probs=config.get_val_datasets('prob'),
        epoch_size=config.get_val('epoch_size'),
        batch_size=config.get_val('batch_size'),
        drop_last=False
    )
    return train_loader, val_loader


def main(config):
    tokenizer = get_tokenizer(config.get('tokenizer'), config)
    os.makedirs(config.get('save_dir'), exist_ok=True)
    train_loader, val_loader = get_loaders(tokenizer, config)

    model = get_model(config.get("model"), tokenizer.get_num_chars(), config)
    if config.get('pretrain_path'):
        states = load_pretrain_model(config.get('pretrain_path'), model)
        model.load_state_dict(states)
        print('Load pretrained model')
    model.to(DEVICE)

    criterion = get_criterion(config.get("criterion")).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        epochs=config.get('num_epochs'),
        steps_per_epoch=len(train_loader),
        max_lr=0.001,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=10 ** 5
    )
    weight_limit_control = FilesLimitControl()
    best_acc = -np.inf

    acc_avg, cer_avg = val_loop(val_loader, model, tokenizer, DEVICE)
    # Collecting statistics
    train_loss_history = np.array([])
    train_cer_history = np.array([cer_avg])
    valid_cer_history = np.array([cer_avg])
    for epoch in range(config.get('num_epochs')):
        loss_avg, cer_avg = train_loop(train_loader, model, criterion, optimizer,
                                       epoch, scheduler, tokenizer, DEVICE)
        train_loss_history = np.append(train_loss_history, loss_avg)
        train_cer_history = np.append(train_cer_history, cer_avg)

        acc_avg, cer_avg = val_loop(val_loader, model, tokenizer, DEVICE)
        valid_cer_history = np.append(valid_cer_history, cer_avg)
        if acc_avg > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(
                config.get('save_dir'), f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')
            weight_limit_control(model_save_path)

    print_plot(train_loss_history, train_cer_history, valid_cer_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='configs/ocr_config.json',
                        help='Path to config.json.')
    args = parser.parse_args()

    print("INFO: DEVICE is " + str(DEVICE))
    main(config=Config(args.config_path))
