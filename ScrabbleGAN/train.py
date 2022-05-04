import argparse
import logging
import os
import random
import shutil
import sys
from importlib import import_module
from itertools import cycle

import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from data_loader.data_generator import DataLoader
from gan_utils.data_utils import *
from gan_utils.training_utils import ModelCheckpoint
from losses_and_metrics import loss_functions

level = logging.INFO
format_log = f"[{os.getpid():04d}] %(asctime)s %(levelname).1s: %(message)s"

os.makedirs('outputs', exist_ok=True)
handlers = [logging.FileHandler('outputs/output.log', mode='w', encoding='utf-8'),
            logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(level=level, format=format_log, handlers=handlers)


def init_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_train_loader(config, data_pkl_path, rank, world_size, worker_count):
    data_loader = DataLoader(config, data_pkl_path)
    data_loader = data_loader.create_train_loader(rank, world_size, max(round(worker_count / world_size), 1))
    return data_loader


class Trainer:
    def __init__(self, config, args, rank: int, local_rank: int, world_size: int):
        self.config = config

        self.local_rank = int(local_rank)
        self.rank = int(rank)
        self.world_size = int(world_size)
        assert 0 <= self.local_rank <= self.rank < self.world_size

        self.terminal_width = shutil.get_terminal_size((80, 20)).columns
        if self.local_rank == 0:
            logging.info(f' Loading Data '.center(self.terminal_width, '*'))

        self.is_unlabeled_data = False
        if args.unlabeled_pkl_path:
            self.is_unlabeled_data = True
            if self.local_rank == 0:
                logging.info("Using unlabeled data to train discriminator")

        self.labeled_loader = get_train_loader(config, args.data_pkl_path,
                                               self.rank, self.world_size,
                                               self.config.worker_count)
        self.num_chars = len(self.labeled_loader.dataset.char_map)
        if self.is_unlabeled_data:
            self.unlabeled_loader = get_train_loader(config, args.unlabeled_pkl_path,
                                                     self.rank, self.world_size,
                                                     self.config.worker_count)

        # Model
        if self.local_rank == 0:
            logging.info(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('gan_models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config, self.labeled_loader.dataset.char_map, args.lexicon_path)

        if self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel
            from torch.nn import SyncBatchNorm
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model.R = DistributedDataParallel(self.model.R,
                                                   # find_unused_parameters=True,
                                                   device_ids=[self.local_rank],
                                                   output_device=self.local_rank)
            self.model.G = DistributedDataParallel(self.model.G,
                                                   # find_unused_parameters=True,
                                                   device_ids=[self.local_rank],
                                                   output_device=self.local_rank)
            self.model.D = DistributedDataParallel(self.model.D,
                                                   find_unused_parameters=True,
                                                   device_ids=[self.local_rank],
                                                   output_device=self.local_rank)
        else:
            self.model.to(self.config.device)
        if self.local_rank == 0:
            logging.info(f'{self.model}\n')

        self.word_map = WordMap(self.labeled_loader.dataset.char_map)

        # Loss, Optimizer and LRScheduler
        self.G_criterion = getattr(loss_functions, self.config.g_loss_fn)('G')
        self.D_criterion = getattr(loss_functions, self.config.d_loss_fn)('D')
        self.R_criterion = getattr(loss_functions, self.config.r_loss_fn)()
        self.G_optimizer = torch.optim.Adam(self.model.G.parameters(),
                                            lr=self.config.g_lr * self.world_size, betas=self.config.g_betas)
        self.D_optimizer = torch.optim.Adam(self.model.D.parameters(),
                                            lr=self.config.d_lr * self.world_size, betas=self.config.d_betas)
        self.R_optimizer = torch.optim.Adam(self.model.R.parameters(),
                                            lr=self.config.r_lr * self.world_size, betas=self.config.r_betas)
        self.optimizers = [self.G_optimizer, self.D_optimizer, self.R_optimizer]
        self.max_grad_norm = dict()

        # Use a linear learning rate decay but start the decay only after specified number of epochs
        def lr_decay_lambda(epoch):
            return (1. - (1. / self.config.epochs_lr_decay)) \
                if epoch > (epoch - self.config.epochs_lr_decay - 1) else 1.
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_decay_lambda) for opt in self.optimizers]

        self.start_epoch = 1
        # Load checkpoint if training is to be resumed
        self.model_checkpoint = ModelCheckpoint(config=self.config, weight_dir="./outputs/weights")
        if args.pretrain_path:
            self.model, self.optimizers, self.schedulers, self.start_epoch = \
                self.model_checkpoint.load(
                    self.model, args.pretrain_path, self.optimizers, self.schedulers, load_only_R=args.load_only_R)
            self.G_optimizer, self.D_optimizer, self.R_optimizer = self.optimizers
            logging.info(f'Resuming model training from epoch {self.start_epoch}')

        # logging
        self.writer = SummaryWriter(f'logs')

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Source - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/fd29199c33bd95704690aaa16f238a4f8e74762c/models/base_model.py
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_G(self):
        """Completes forward, backward, and optimize for G"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch off backpropagation for R and D
        self.set_requires_grad([self.model.D, self.model.R], False)

        # Generator loss will be determined by the evaluation of generated image by discriminator and recognizer
        pred_D_fake = self.model.D(self.model.fake_img)
        pred_R_fake = self.model.R(self.model.fake_img).permute(1, 0, 2)  # [w, b, num_chars]

        self.loss_G = self.G_criterion(pred_D_fake)
        self.loss_R_fake = self.R_criterion(pred_R_fake, self.model.fake_y,
                                            torch.ones(pred_R_fake.size(1)).int() * pred_R_fake.size(0),
                                            self.model.fake_y_lens)
        self.loss_R_fake = torch.mean(self.loss_R_fake[~torch.isnan(self.loss_R_fake)])

        # the below part has been mostly copied from -
        # https://github.com/amzn/convolutional-handwriting-gan/blob/2cfbc794cca299445e5ba070c8634b6cd1a84261/models
        # /ScrabbleGAN_baseModel.py#L345

        if hasattr(self.model.G, "no_sync"):
            with self.model.G.no_sync():
                self._balance_G_loss()
        else:
            self._balance_G_loss()

        with torch.no_grad():
            self.loss_G_total.backward()

        self._clip_gradients('G', self.model.G)
        self.G_optimizer.step()
        self.G_optimizer.zero_grad()

    def _balance_G_loss(self):
        grad_fake_R = torch.autograd.grad(self.loss_R_fake, self.model.fake_img,
                                          create_graph=False, retain_graph=True)[0].detach()
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.model.fake_img,
                                            create_graph=False, retain_graph=True)[0].detach()
        if self.config.grad_balance:
            epsilon = 1e-24
            scale = torch.sqrt(torch.sum(grad_fake_adv ** 2) / (epsilon + torch.sum(grad_fake_R ** 2)))
            scale = torch.clip(scale, min=self.config.min_grad_scale, max=self.config.max_grad_scale)
        else:
            scale = 1

        self.loss_grad_fake_R = 10 ** 6 * torch.mean(grad_fake_R ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        self.loss_G_total = self.loss_G + self.config.grad_alpha * scale * self.loss_R_fake

    def optimize_D_unlabeled(self):
        """Completes forward, backward, and optimize for D on unlabeled data"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch on backpropagation for R and D
        self.set_requires_grad([self.model.D], True)
        pred_D_fake = self.model.D(self.model.fake_img.detach())
        pred_D_real = self.model.D(self.real_unlabeled_img.detach())
        # we will now calculate discriminator loss for both real and fake images
        self.loss_D_fake = self.D_criterion(pred_D_fake, 'fake')
        self.loss_D_real = self.D_criterion(pred_D_real, 'real')
        self.loss_D = self.loss_D_fake + self.loss_D_real

        self.loss_D.backward()

        self._clip_gradients('D', self.model.D)
        self.D_optimizer.step()
        self.D_optimizer.zero_grad()

    def optimize_D_R(self):
        """Completes forward, backward, and optimize for D and R"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch on backpropagation for R and D
        self.set_requires_grad([self.model.D, self.model.R], True)

        pred_D_fake = self.model.D(self.model.fake_img.detach())
        pred_D_real = self.model.D(self.real_img.detach())

        # we will now calculate discriminator loss for both real and fake images
        self.loss_D_fake = self.D_criterion(pred_D_fake, 'fake')
        self.loss_D_real = self.D_criterion(pred_D_real, 'real')
        self.loss_D = self.loss_D_fake + self.loss_D_real

        # recognizer
        self.pred_R_real = self.model.R(self.real_img).permute(1, 0, 2)  # [w, b, num_chars]
        self.loss_R_real = self.R_criterion(self.pred_R_real, self.real_y,
                                            torch.ones(self.pred_R_real.size(1)).int() * self.pred_R_real.size(0),
                                            self.real_y_lens)
        self.loss_R_real = torch.mean(self.loss_R_real[~torch.isnan(self.loss_R_real)])

        self.loss_D_and_R = self.loss_D + self.loss_R_real

        self.loss_D_and_R.backward()

        self._clip_gradients('D', self.model.D)
        self._clip_gradients('R', self.model.R)
        self.D_optimizer.step()
        self.R_optimizer.step()
        self.D_optimizer.zero_grad()
        self.R_optimizer.zero_grad()

    def _clip_gradients(self, name, model):
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.max_grad_norm)
        self.max_grad_norm[name] = max(self.max_grad_norm.get(name, 0.), float(total_norm))

    def train(self):
        logging.info(f' Training '.center(self.terminal_width, '*'))

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            logging.info(f' Epoch [{epoch}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
            self.model.train()

            if self.world_size > 1 and epoch > self.start_epoch:
                import torch.distributed
                for p in self.model.parameters():
                    torch.distributed.broadcast(p, 0)

                for o in self.optimizers:
                    for _, v in sorted(o.state_dict()['state'].items()):
                        if isinstance(v, torch.Tensor):
                            torch.distributed.broadcast(v, 0)

            train_loader = self.labeled_loader
            len_loader = len(self.labeled_loader)
            if self.is_unlabeled_data:
                if len(self.unlabeled_loader) > len(self.labeled_loader):
                    train_loader = zip(cycle(self.labeled_loader), self.unlabeled_loader)
                else:
                    train_loader = zip(self.labeled_loader, cycle(self.unlabeled_loader))

            progbar = tqdm(train_loader, total=len_loader, disable=None)
            losses_G, losses_D, losses_D_real, losses_D_fake = [], [], [], []
            losses_R_real, losses_R_fake, grads_fake_R, grads_fake_adv = [], [], [], []
            for i, batch_items in enumerate(progbar):
                if self.is_unlabeled_data:
                    batch_items, data_unlabeled = batch_items
                    self.real_unlabeled_img = data_unlabeled['img'].to(self.config.device)

                self.real_img = batch_items['img'].to(self.config.device)
                self.real_y = batch_items['label'].to(self.config.device)
                self.real_y_one_hot = F.one_hot(batch_items['label'], self.num_chars).to(self.config.device)
                self.real_y_lens = batch_items['label_len'].to(self.config.device)

                # Forward + Backward + Optimize G
                if (i % self.config.train_gen_steps) == 0:
                    # optimize generator
                    self.optimize_G()

                # Forward + Backward + Optimize D and R
                self.optimize_D_R()

                if self.is_unlabeled_data:
                    self.optimize_D_unlabeled()

                # save losses
                losses_G.append(self.loss_G.cpu().data.numpy())
                losses_D.append(self.loss_D.cpu().data.numpy())
                losses_D_real.append(self.loss_D_real.cpu().data.numpy())
                losses_D_fake.append(self.loss_D_fake.cpu().data.numpy())
                losses_R_real.append(self.loss_R_real.cpu().data.numpy())
                losses_R_fake.append(self.loss_R_fake.cpu().data.numpy())
                grads_fake_R.append(self.loss_grad_fake_R.cpu().data.numpy())
                grads_fake_adv.append(self.loss_grad_fake_adv.cpu().data.numpy())

                progbar.set_description("G = %0.3f, D = %0.3f, R_real = %0.3f, R_fake = %0.3f,  " %
                                        (np.mean(losses_G), np.mean(losses_D),
                                         np.mean(losses_R_real), np.mean(losses_R_fake)))

            logging.info(f'G = {np.mean(losses_G):.3f}, D = {np.mean(losses_D):.3f}, '
                         f'R_real = {np.mean(losses_R_real):.3f}, R_fake = {np.mean(losses_R_fake):.3f}')
            logging.info(f'grad_fake_R = {np.mean(grads_fake_R):.4g}, grad_G = {np.mean(grads_fake_adv):.4g}')
            if self.local_rank == 0:
                logging.info(f'Max grad norm: ' +
                             ', '.join(f'{n} = {v:.4g}' for n, v in sorted(self.max_grad_norm.items())))
            self.max_grad_norm.clear()

            # Save one generated fake image from last batch
            if self.local_rank == 0:
                img = self.model.fake_img.cpu().data.numpy()[0]
                normalized_img = ((img + 1) * 255 / 2).astype(np.uint8)
                normalized_img = np.moveaxis(normalized_img, 0, -1)
                os.makedirs('./outputs/images', exist_ok=True)
                cv2.imwrite(f'./outputs/images/epoch_{epoch}_fake_img.png', normalized_img)

            # Print Recognizer prediction for 4 (or batch size) real images from last batch
            num_imgs = 4 if self.config.batch_size >= 4 else self.config.batch_size
            labels = self.word_map.decode(self.real_y[:num_imgs].cpu().numpy())
            preds = self.word_map.recognizer_decode(self.pred_R_real.max(2)[1].permute(1, 0)[:num_imgs].cpu().numpy())
            logging.info('Recognizer predictions for real images:')
            max_len_label = max([len(i) for i in labels])
            for lab, pred in zip(labels, preds):
                logging.info(f'Actual: {lab:<{max_len_label + 2}}|  Predicted: {pred}')

            # Print Recognizer prediction for 4 (or batch size) fake images from last batch
            logging.info('Recognizer predictions for fake images:')
            labels = self.word_map.decode(self.model.fake_y[:num_imgs].cpu().numpy())
            preds_R_fake = self.model.R(self.model.fake_img).permute(1, 0, 2).max(2)[1].permute(1, 0)
            preds = self.word_map.recognizer_decode(preds_R_fake[:num_imgs].cpu().numpy())
            max_len_label = max([len(i) for i in labels])
            for lab, pred in zip(labels, preds):
                logging.info(f'Actual: {lab:<{max_len_label + 2}}|  Predicted: {pred}')

            # Change learning rate according to scheduler
            for sch in self.schedulers:
                sch.step()

            # save checkpoint after every 5 epochs
            if epoch % 5 == 0 and self.local_rank == 0:
                self.model_checkpoint.save(self.model, epoch, self.G_optimizer, self.D_optimizer, self.R_optimizer,
                                           *self.schedulers)

            # write logs
            self.writer.add_scalar(f'loss_G', np.mean(losses_G), epoch * i)
            self.writer.add_scalar(f'loss_D/fake', np.mean(losses_D_fake), epoch * i)
            self.writer.add_scalar(f'loss_D/real', np.mean(losses_D_real), epoch * i)
            self.writer.add_scalar(f'loss_R/fake', np.mean(losses_R_fake), epoch * i)
            self.writer.add_scalar(f'loss_R/real', np.mean(losses_R_real), epoch * i)
            self.writer.add_scalar(f'grads/fake_R', np.mean(grads_fake_R), epoch * i)
            self.writer.add_scalar(f'grads/fake_adv', np.mean(grads_fake_adv), epoch * i)

        self.writer.close()


def setup_distributed_rank(local_rank: int):
    import torch.distributed
    print(f"Initializing NCCL local_rank={local_rank}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"Using {torch.distributed.get_backend()}, rank={rank}, world_size={world_size}")
    return rank, world_size


def cleanup_distributed_rank():
    import torch.distributed
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pkl_path", required=True, type=str,
                        help="Path to the pickle processed data")
    parser.add_argument("--unlabeled_pkl_path", type=str, default='',
                        help="Path to the pickle processed unlabeled data")
    parser.add_argument("--pretrain_path", type=str, default='',
                        help="Path to the pretrain model weights")
    parser.add_argument("--load_only_R", action='store_true',
                        help="To load only Recognizer from model weights")
    parser.add_argument("--lexicon_path", action='append', required=True,
                        type=str, help="Path to the lexicon txt. Can be passed "
                                       "multiple times")
    parser.add_argument("--distributed", action="store_true", help="run in distributed job")
    args = parser.parse_args()

    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank, world_size = setup_distributed_rank(local_rank)
    else:
        local_rank, rank, world_size = 0, 0, 1

    init_seed(local_rank)

    try:
        trainer = Trainer(Config, args, local_rank, rank, world_size)
        trainer.train()
    finally:
        if args.distributed:
            cleanup_distributed_rank()


if __name__ == "__main__":
    main()
