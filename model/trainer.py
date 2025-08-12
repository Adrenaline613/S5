import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

import webbrowser
import os
import time
from pathlib import Path
import logging

from model.metric import *
from model.postprocessing import process_events


class TrainerCl:
    """
    Trainer class for training and validating PyTorch models.

    Attributes:
        name (str): Name of the experiment.

        model (nn.Module): The PyTorch model to be trained.

        train_loader (DataLoader): DataLoader for the training dataset.

        val_loader (DataLoader): DataLoader for the validation dataset.

        criterion (nn.Module): Loss function.

        optimizer (optim.Optimizer): Optimizer for training.

        num_epochs (int): Number of epochs to train the model.

        cross_val_idx (int): Index of the current cross-validation fold.

        scheduler (optim.lr_scheduler, optional): Learning rate scheduler.
            default None.

        device (str): Device to run the model on ('cuda' or 'cpu').
            default 'auto'.

        ckpt_save_mode (str): Mode for saving model weights.
            'best' or 'all'. if 'best', only the best epoch weight is saved. else all epochs are saved.
            default 'best'.

        print_interval (int): Interval for printing batch average loss.
            default 10.

        patience (int, optional): Patience for early stopping.
            when None, early stopping is disabled.
            default None.

        early_stop_metric (str): Metric to monitor for early stopping.
            when patience is not None, this metric is used to determine improvement.
            default 'val_loss'.

        writer (SummaryWriter, optional): TensorBoard SummaryWriter.
            when True, logs are written to tensorboard. path is 'output/runs/self.name'.
            default True.

    """

    def __init__(
            self,
            name: str,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            num_epochs: int,
            cross_val_idx: int,
            scheduler: optim.lr_scheduler = None,
            device = 'auto',
            ckpt_save_mode: str = 'best',
            print_interval: int = 10,
            patience: int = None,
            early_stop_metric: str = 'val_loss',
            writer: bool = True,
    ):

        self.name = name
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.cross_val_idx = cross_val_idx
        self.scheduler = scheduler
        assert ckpt_save_mode in ['best', 'all'], 'Invalid checkpoint save mode'
        self.ckpt_save_mode = ckpt_save_mode
        self.patience = patience
        self.print_interval = print_interval
        assert early_stop_metric in ['val_loss'], 'Invalid early stopping metric'
        self.early_stop_metric = early_stop_metric
        self.best_metric = float('inf') if early_stop_metric == 'val_loss' else float('-inf')
        self.best_epoch = 0
        self.early_stop_counter = 0
        self.tensorboard_dir = os.path.join('tensorboard', name, f'fold_{cross_val_idx}')
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir) if writer else None

        root_dir = Path(__file__).parent.parent
        self.save_dir = os.path.join(root_dir, 'weight', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger(f'training_logger_fold_{self.cross_val_idx}')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.save_dir, f'val_result_fold_{self.cross_val_idx}.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
        return logger

    def train_one_epoch(self, epoch):
        print('\033[31m' + 20 * '-' + 'Training...' + 20 * '-' + '\033[0m')

        self.model.train()

        train_loss = 0.0
        start_time = time.time()

        for step, inputs in enumerate(self.train_loader):
            loss = 0
            b_num = inputs[0].size(0)

            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(self.device)
            outputs = self.model(inputs)
            f1, f2 = torch.split(outputs, [b_num, b_num], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss += self.criterion(features)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if step % self.print_interval == 0:
                print(f'Step {step}/{len(self.train_loader)}, Batch Avg Loss: {loss.item():.4f}')

        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch} completed in {epoch_time:.2f}s')

        epoch_loss = train_loss / len(self.train_loader)
        print(f'Train Loss: {epoch_loss:.4f}')

        return epoch_loss

    def validate_one_epoch(self):
        print('\033[31m' + 19 * '-' + 'Validating...' + 19 * '-' + '\033[0m')

        self.model.eval()

        val_loss = 0.0

        for step, inputs in enumerate(self.val_loader):
            loss = 0
            b_num = inputs[0].size(0)

            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(self.device)
            outputs = self.model(inputs)
            f1, f2 = torch.split(outputs, [b_num, b_num], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss += self.criterion(features)

            val_loss += loss.item()

            if step % self.print_interval == 0:
                print(f'Step {step}/{len(self.val_loader)}, Batch Avg Loss: {loss.item():.4f}')

        epoch_loss = val_loss / len(self.val_loader)
        print(f'Validation Loss: {epoch_loss:.4f}')

        return epoch_loss

    def save_model(self, epoch, val_loss):

        if self.ckpt_save_mode == 'best':
            ckpt_save_name = os.path.join(self.save_dir, f'fold{self.cross_val_idx}_best.ckpt')
        else:
            ckpt_save_name = os.path.join(self.save_dir, f'fold{self.cross_val_idx}_epoch{epoch}.ckpt')

        torch.save({
            'cross_val_idx': self.cross_val_idx,
            'epoch': epoch,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
        }, ckpt_save_name)

    def fit(self, num_epochs=None):

        epochs = num_epochs if num_epochs is not None else self.num_epochs

        if self.writer:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', self.tensorboard_dir])
            url = tb.launch()
            webbrowser.open(url)

        self.logger.info(f'fold: {self.cross_val_idx}')

        for e in range(epochs):
            epoch = e + 1
            print('\033[31m' + '\n' + 20 * '=' + f'Epoch {epoch}/{epochs}' + 20 * '=' + '\033[0m')

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch()

            if self.scheduler:
                self.scheduler.step()
                if self.writer:
                    self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)

            if self.ckpt_save_mode == 'all':
                self.save_model(epoch, val_loss)

            self.logger.info(f'Epoch {epoch}:\t val_loss: {val_loss:.4f}')

            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)

            if self.patience is not None:
                metric = val_loss

                if metric < self.best_metric:
                    self.best_metric = metric
                    self.early_stop_counter = 0
                    self.best_epoch = epoch
                    if self.ckpt_save_mode == 'best':
                        self.save_model(epoch, val_loss)
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print("Early stopping triggered:")
                        print(f"Best {self.early_stop_metric} in epoch {self.best_epoch}: {self.best_metric}")
                        break

        else:
            print("Run out of epochs")
            print(f"Best {self.early_stop_metric} in epoch {self.best_epoch}: {self.best_metric}")

        self.logger.info(f'Best {self.early_stop_metric} in epoch {self.best_epoch}: {self.best_metric}\n')

        if self.writer:
            self.writer.close()
            print(f'Tensorboard has been closed, if you want to reopen it, please run the following command in terminal'
                  f': tensorboard --logdir={self.tensorboard_dir}\n')

        return self.best_epoch, self.best_metric


class TrainerSp:
    """
    Trainer class for training and validating PyTorch models.

    Attributes:
        name (str): Name of the experiment.

        model (nn.Module): The PyTorch model to be trained.

        train_loader (DataLoader): DataLoader for the training dataset.

        val_loader (DataLoader): DataLoader for the validation dataset.

        criterion (nn.Module): Loss function.

        optimizer (optim.Optimizer): Optimizer for training.

        num_epochs (int): Number of epochs to train the model.

        cross_val_idx (int): Index of the current cross-validation fold.

        scheduler (optim.lr_scheduler, optional): Learning rate scheduler.
            default None.

        device (str): Device to run the model on ('cuda' or 'cpu').
            default 'auto'.

        ckpt_save_mode (str): Mode for saving model weights.
            'best' or 'all'. if 'best', only the best epoch weight is saved. else all epochs are saved.
            default 'best'.

        print_interval (int): Interval for printing .
            default 10.

        patience (int, optional): Patience for early stopping.
            when None, early stopping is disabled.
            default None.

        early_stop_metric (str): Metric to monitor for early stopping.
            when patience is not None, this metric is used to determine improvement.
            'acc': accuracy by point.
            'f1': f1 score by point.
            '2-f1': f1 score by spindle (IoU=0.2).
            '2-f1m': mean f1 score by spindle (IoU=0.2).
            default '2-f1m'.

        overlap_step (float): Step size for overlap threshold.
            calculate the metric for each IoU threshold.
            default 0.05.

        writer (SummaryWriter, optional): TensorBoard SummaryWriter.
            when True, logs are written to tensorboard. path is 'output/runs/self.name'.
            default True.

        postprocess (bool): Whether to apply postprocessing to the model output.
            default False.
    """

    def __init__(
            self,
            name: str,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            num_epochs: int,
            cross_val_idx: int,
            test_loader: DataLoader = None,
            scheduler: optim.lr_scheduler = None,
            device: str = 'auto',
            ckpt_save_mode: str = 'best',
            print_interval: int = 10,
            patience: int = None,
            early_stop_metric: str = '2-mf1',
            overlap_step: float = 0.05,
            writer: bool = True,
            postprocess: bool = False
    ):

        self.name = name
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.cross_val_idx = cross_val_idx
        assert ckpt_save_mode in ['best', 'all'], 'Invalid checkpoint save mode'
        self.ckpt_save_mode = ckpt_save_mode
        self.patience = patience
        self.print_interval = print_interval
        assert early_stop_metric in ['acc', 'f1', '2-f1', '2-f1m'], 'Invalid early stopping metric'
        self.early_stop_metric = early_stop_metric
        self.best_metric = float('inf') if early_stop_metric == 'val_loss' else float('-inf')
        self.best_epoch = 0
        self.early_stop_counter = 0
        self.overlap_thresholds, step = np.linspace(0., 1., int(1 / overlap_step) + 1, retstep=True)
        self.tensorboard_dir = os.path.join('tensorboard', name, f'fold_{cross_val_idx}')
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir) if writer else None
        self.postprocess = postprocess

        root_dir = Path(__file__).parent.parent
        self.save_dir = os.path.join(root_dir, 'weight', self.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger(f'training_logger_fold_{self.cross_val_idx}')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.save_dir, f'val_result_fold_{self.cross_val_idx}.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
        return logger

    def train_one_epoch(self, epoch):
        print('\033[31m' + 20 * '-' + 'Training...' + 20 * '-' + '\033[0m')

        train_loss = 0.0

        # [TN, FP, FN, TP], to store the sum of all batches
        all_samples_metric = [0, 0, 0, 0]
        # [spindles_pred, spindles_true, true_positives, iou_sum], to store the sum of all batches
        all_overlap_metric = [0, 0, np.zeros_like(self.overlap_thresholds, dtype=int),
                              np.zeros_like(self.overlap_thresholds)]

        self.model.train()

        start_time = time.time()

        for step, (inputs, labels) in enumerate(self.train_loader):
            # If batch_size=1, batch norm cannot be performed, so it needs to be skipped
            if inputs.size(0) == 1:
                print(f'Skip batch {step}/{len(self.train_loader)} due to batch size 1')
                continue

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)

            # Samples metric, one batch
            batch_samples_metric = get_samples_metric_num_batch(preds, labels)
            all_samples_metric = [x + y for x, y in zip(all_samples_metric, batch_samples_metric)]

            # Overlap metric, one batch
            batch_overlap_metric = get_overlap_metric_num_batch(preds, labels, self.overlap_thresholds)
            all_overlap_metric = [x + y for x, y in zip(all_overlap_metric, batch_overlap_metric)]

            if step % self.print_interval == 0:
                print(f'Step {step+1}/{len(self.train_loader)}, Batch Avg Loss: {loss.item():.4f}')

        epoch_time = time.time() - start_time
        print(f'\nEpoch {epoch} completed in {epoch_time:.2f}s')

        epoch_loss = train_loss / len(self.train_loader.dataset)

        # Samples metric, all batches
        # s_metric_result = (acc, pre, sen, spe, f1)
        s_metric_result = samples_metrics(all_samples_metric, print_table=False)
        # Overlap metric, all batches
        # o_metric_result = (pre, rec, f1, mf1, miou)
        o_metric_result = overlap_metrics(all_overlap_metric, self.overlap_thresholds, print_threshold=0.2,
                                          print_table=False)

        return epoch_loss, s_metric_result, o_metric_result

    def validate_one_epoch(self, mode='val'):
        assert mode in ['val', 'test'], 'Invalid mode'
        if mode == 'val':
            dataloader = self.val_loader
            print('\033[31m' + 19 * '-' + 'Validating...' + 19 * '-' + '\033[0m')
        else:
            dataloader = self.test_loader
            print('\033[31m' + 19 * '-' + 'Testing...' + 19 * '-' + '\033[0m')

        self.model.eval()
        running_loss = 0.0

        # [TN, FP, FN, TP], to store the sum of all batches
        all_samples_metric = [0, 0, 0, 0]
        # [spindles_pred, spindles_true, true_positives, iou_sum], to store the sum of all batches
        all_overlap_metric = [0, 0, np.zeros_like(self.overlap_thresholds, dtype=int),
                              np.zeros_like(self.overlap_thresholds)]

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)

                if self.postprocess:
                    preds = preds.cpu().numpy()
                    for idx, pred in enumerate(preds):
                        preds[idx] = process_events(pred, 100)

                # Samples metric, one batch
                batch_samples_metric = get_samples_metric_num_batch(preds, labels)
                all_samples_metric = [x + y for x, y in zip(all_samples_metric, batch_samples_metric)]

                # Overlap metric, one batch
                batch_overlap_metric = get_overlap_metric_num_batch(preds, labels, self.overlap_thresholds)
                all_overlap_metric = [x + y for x, y in zip(all_overlap_metric, batch_overlap_metric)]

        epoch_loss = running_loss / len(self.val_loader.dataset)

        # Samples metric, all batches
        # s_metric_result = (acc, pre, sen, spe, f1)
        s_metric_result = samples_metrics(all_samples_metric, print_table=True)
        # Overlap metric, all batches
        # o_metric_result = (pre, rec, f1, mf1, miou)
        o_metric_result = overlap_metrics(all_overlap_metric, self.overlap_thresholds, print_threshold=0.2,
                                          print_table=True)

        return epoch_loss, s_metric_result, o_metric_result

    def save_model(self, epoch, f1, f1_metric2, f1_mean):

        if self.ckpt_save_mode == 'best':
            ckpt_save_name = os.path.join(self.save_dir, f'fold{self.cross_val_idx}_best.ckpt')
        else:
            ckpt_save_name = os.path.join(self.save_dir, f'fold{self.cross_val_idx}_epoch{epoch}.ckpt')

        torch.save({
            'cross_val_idx': self.cross_val_idx,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_f1': f1,
            'val_f1_metric2': f1_metric2,
            'val_f1_mean': f1_mean
        }, ckpt_save_name)

    def fit(self, num_epochs=None):

        epochs = num_epochs if num_epochs is not None else self.num_epochs

        if self.writer:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', self.tensorboard_dir])
            url = tb.launch()
            webbrowser.open(url)

        self.logger.info(f'fold: {self.cross_val_idx}')

        for e in range(epochs):
            epoch = e + 1
            print('\033[31m' + '\n' + 20 * '=' + f'Epoch {epoch}/{epochs}' + 20 * '=' + '\033[0m')

            if e == 20:
                print('test')

            train_loss, train_metric_result, train_metric2_result = self.train_one_epoch(epoch)
            val_loss, val_metric_result, val_metric2_result = self.validate_one_epoch(mode='val')

            if self.test_loader is not None:
                test_loss, test_metric_result, test_metric2_result = self.validate_one_epoch(mode='test')

            if self.scheduler:
                self.scheduler.step()
                if self.writer:
                    self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)

            if self.ckpt_save_mode == 'all':
                self.save_model(epoch, val_metric_result[4], val_metric2_result[2], val_metric2_result[3])

            self.logger.info(f'Epoch {epoch}:\t Precision: {val_metric2_result[0]:.4f},'
                             f'Recall: {val_metric2_result[1]:.4f}, F1: {val_metric2_result[2]:.4f},'
                             f'F1-mean: {val_metric2_result[3]:.4f}, mIoU: {val_metric2_result[4]:.4f}')

            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Samples-Accuracy/train', train_metric_result[0], epoch)
                self.writer.add_scalar('Samples-Accuracy/val', val_metric_result[0], epoch)
                self.writer.add_scalar('Samples-F1/train', train_metric_result[4], epoch)
                self.writer.add_scalar('Samples-F1/val', val_metric_result[4], epoch)
                self.writer.add_scalar('Overlap-F1/train', train_metric2_result[2], epoch)
                self.writer.add_scalar('Overlap-F1/val', val_metric2_result[2], epoch)
                self.writer.add_scalar('Overlap-mF1/train', train_metric2_result[3], epoch)
                self.writer.add_scalar('Overlap-mF1/val', val_metric2_result[3], epoch)
                self.writer.add_scalar('Overlap-mIoU/train', train_metric2_result[4], epoch)
                self.writer.add_scalar('Overlap-mIoU/val', val_metric2_result[4], epoch)

                if self.test_loader is not None:
                    self.writer.add_scalar('Loss/test', test_loss, epoch)
                    self.writer.add_scalar('Samples-Accuracy/test', test_metric_result[0], epoch)
                    self.writer.add_scalar('Samples-F1/test', test_metric_result[4], epoch)
                    self.writer.add_scalar('Overlap-F1/test', test_metric2_result[2], epoch)
                    self.writer.add_scalar('Overlap-mF1/test', test_metric2_result[3], epoch)
                    self.writer.add_scalar('Overlap-mIoU/test', test_metric2_result[4], epoch)

            if self.patience is not None:
                if self.early_stop_metric == 'acc':
                    metric = val_metric_result[0]
                elif self.early_stop_metric == 'f1':
                    metric = val_metric_result[4]
                elif self.early_stop_metric == '2-f1':
                    metric = val_metric2_result[2]
                else:
                    metric = val_metric2_result[3]

                if metric > self.best_metric:
                    self.best_metric = metric
                    self.early_stop_counter = 0
                    self.best_epoch = epoch
                    if self.ckpt_save_mode == 'best':
                        self.save_model(epoch, val_metric_result[4], val_metric2_result[2], val_metric2_result[3])
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        print("Early stopping triggered:")
                        print(f"Best {self.early_stop_metric} in epoch {self.best_epoch}: {self.best_metric}")
                        break

        else:
            print("Run out of epochs")
            print(f"Best {self.early_stop_metric} in epoch {self.best_epoch}: {self.best_metric}")

        self.logger.info(f'Best {self.early_stop_metric} in epoch {self.best_epoch}: {self.best_metric}\n')

        if self.writer:
            self.writer.close()
            print(
                f'Tensorboard has been closed, if you want to reopen it, please run the following command in terminal: tensorboard --logdir={self.tensorboard_dir}\n')

        return self.best_epoch, self.best_metric
