from tqdm import tqdm
import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.loop_helper import (
    val_loop, load_pretrain_model, FilesLimitControl, AverageMeter, sec2min
)

from utils.loop_helper import cer
from utils.dataset import get_data_loader
from utils.transforms import get_train_transforms, get_val_transforms
from utils.tokenizer import Tokenizer
from utils.config import Config
from models.builder import (get_model, get_criterion)

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
    plt.show()


def get_text_from_probs(output, tokenizer):
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    return tokenizer.decode(pred)


def train_loop(data_loader, model, criterion, optimizer, epoch, scheduler, tokenizer):
    torch.cuda.empty_cache()
    loss_avg = AverageMeter()
    cer_avg = AverageMeter()
    start_time = time.time()
    model.train()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, enc_pad_texts, text_lens in tqdm_data_loader:
        model.zero_grad()
        images = images.to(DEVICE)
        batch_size = len(texts)
        output = model(images)
        output_lengths = torch.full(
            size=(output.size(1),),
            fill_value=output.size(0),
            dtype=torch.long
        )
        loss = criterion(output, enc_pad_texts, output_lengths, text_lens)
        loss_avg.update(loss.item(), batch_size)
        cer_avg.update(cer(texts, get_text_from_probs(output, tokenizer)), batch_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        scheduler.step()
    loop_time = sec2min(time.time() - start_time)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print(f'\nEpoch {epoch}, Loss: {loss_avg.avg:.5f}, cer: {cer_avg.avg:.4f}, '
          f'LR: {lr:.7f}, loop_time: {loop_time}')
    return loss_avg.avg, cer_avg.avg


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


def main(args):
    config = Config(args.config_path)
    tokenizer = Tokenizer(config.get('alphabet'))
    os.makedirs(config.get('save_dir'), exist_ok=True)
    train_loader, val_loader = get_loaders(tokenizer, config)

    model = get_model(config.get("model"), number_class_symbols=tokenizer.get_num_chars())
    if config.get('pretrain_path'):
        states = load_pretrain_model(config.get('pretrain_path'), model)
        model.load_state_dict(states)
        print('Load pretrained model')
    model.to(DEVICE)

    criterion = get_criterion(config.get("criterion"))
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
                                       epoch, scheduler, tokenizer)
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
                        default='ocr_config.json',
                        help='Path to config.json.')
    args = parser.parse_args()

    print("INFO: DEVICE is " + str(DEVICE))
    main(args)
