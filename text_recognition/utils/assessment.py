import torch
import math
import time
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from utils.metrics import get_accuracy, wer, cer
from utils.predictor import predict, get_text_from_probs


def train_loop(data_loader, model, criterion, optimizer, scheduler, tokenizer, device,
               epoch: int, writer: SummaryWriter, fine_tuning: bool = True):
    torch.cuda.empty_cache()
    loss_avg = AverageMeter()
    cer_avg = AverageMeter()
    grad_norm_avg = AverageMeter()
    start_time = time.time()
    model.train()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, enc_pad_texts, text_lens in tqdm_data_loader:
        model.zero_grad()
        images = images.to(device)
        enc_pad_texts = enc_pad_texts.to(device)
        batch_size = len(texts)
        output = model(images, fine_tuning)
        loss = criterion(output, enc_pad_texts, text_lens)
        loss_avg.update(loss.item(), batch_size)
        cer_avg.update(cer(texts, get_text_from_probs(output, tokenizer)), batch_size)
        loss.backward()
        grad_norm_avg.update(torch.nn.utils.clip_grad_norm_(model.parameters(), 2))
        optimizer.step()
        scheduler.step()
    loop_time = sec2min(time.time() - start_time)

    # Saving statistics
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    writer.add_scalar('Grad/train', grad_norm_avg.avg, epoch)
    writer.add_scalar('Loss/train', loss_avg.avg, epoch)
    writer.add_scalar('CER/train', cer_avg.avg, epoch)
    writer.add_scalar('LR/train', lr, epoch)
    print(f'\nEpoch {epoch}, Loss: {loss_avg.avg:.5f}, cer: {cer_avg.avg:.4f}, '
          f'LR: {lr:.7f}, loop_time: {loop_time}')
    return loss_avg.avg, cer_avg.avg


def val_loop(data_loader, model, tokenizer, device, epoch: int = 0, writer: SummaryWriter = None):
    acc_avg = AverageMeter()
    wer_avg = AverageMeter()
    cer_avg = AverageMeter()
    start_time = time.time()
    model.eval()
    tqdm_data_loader = tqdm(data_loader, total=len(data_loader), leave=False)
    for images, texts, _, _ in tqdm_data_loader:
        batch_size = len(texts)
        text_preds = predict(images, model, tokenizer, device)
        acc_avg.update(get_accuracy(texts, text_preds), batch_size)
        wer_avg.update(wer(texts, text_preds), batch_size)
        cer_avg.update(cer(texts, text_preds), batch_size)

    loop_time = sec2min(time.time() - start_time)

    # Saving statistics
    if writer is not None:
        writer.add_scalar('Accuracy/test', acc_avg.avg, epoch)
        writer.add_scalar('WER/test', wer_avg.avg, epoch)
        writer.add_scalar('CER/test', cer_avg.avg, epoch)
    print(f'\nValidation\t'
          f'acc: {acc_avg.avg:.4f}, '
          f'wer: {wer_avg.avg:.4f}, '
          f'cer: {cer_avg.avg:.4f}, '
          f'loop_time: {loop_time}')
    return acc_avg.avg, cer_avg.avg


def sec2min(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
