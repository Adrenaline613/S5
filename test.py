import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

from model.metric import *
from model.postprocessing import process_events
from model.net import S3Net


class TestDataset(Dataset):
    def __init__(self, np_dataset):

        x = np.load(np_dataset[0])["segments"]
        y = np.load(np_dataset[0])["labels"]

        for np_file in np_dataset[1:]:
            x = np.concatenate((x, np.load(np_file)["segments"]), axis=0)
            y = np.concatenate((y, np.load(np_file)["labels"]), axis=0)

        x = x.astype(np.float32)
        y = y.astype(np.int8)

        labels, counts = np.unique(y, return_counts=True)
        print('Labels:', labels, 'Counts:', counts)
        print('epochs:', x.shape[0])

        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(1, 0, 2)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        print('ok')

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def test_model(fold, ckpt_dir, data_dir):

    ckpt_paths = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    ckpt_paths.sort()

    if fold == 'all':
        ckpt_paths = ckpt_paths[:]
    elif fold in [1, 2, 3, 4, 5, 6]:
        ckpt_paths = [ckpt_paths[fold-1]]
    else:
        raise ValueError('fold must be all, 1, 2, 3, 4, 5, 6')

    result_sample_list = []
    result_overlap_list = []

    for ckpt_path in ckpt_paths:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = S3Net(in_channels=1, num_classes=2, training_mode='scratch')
        model = model.to(device)
        model_checkpoint = torch.load(ckpt_path)
        model.load_state_dict(model_checkpoint['model_state_dict'])

        data_path_list = glob.glob(os.path.join(data_dir, '*.npz'))
        dataset = TestDataset(data_path_list)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)

        model.eval()
        # [TN, FP, FN, TP], to store the sum of all batches

        overlap_step: float = 0.05
        overlap_thresholds, step = np.linspace(0., 1., int(1 / overlap_step) + 1, retstep=True)
        all_samples_metric = [0, 0, 0, 0]
        # [spindles_pred, spindles_true, true_positives, iou_sum], to store the sum of all batches
        all_overlap_metric = [0, 0, np.zeros_like(overlap_thresholds, dtype=int),
                              np.zeros_like(overlap_thresholds)]

        with torch.no_grad():

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                preds = torch.argmax(outputs, dim=1)

                preds = preds.cpu().numpy()
                for idx, pred in enumerate(preds):
                    preds[idx] = process_events(pred, 100)

                # Samples metric, one batch
                batch_samples_metric = get_samples_metric_num_batch(preds, labels)
                all_samples_metric = [x + y for x, y in zip(all_samples_metric, batch_samples_metric)]

                # Overlap metric, one batch
                batch_overlap_metric = get_overlap_metric_num_batch(preds, labels, overlap_thresholds)
                all_overlap_metric = [x + y for x, y in zip(all_overlap_metric, batch_overlap_metric)]

        inputs = inputs.cpu().numpy()
        labels = labels.cpu().numpy()

        # Samples metric, all batches
        # s_metric_result = (acc, pre, sen, spe, f1)
        s_metric_result = samples_metrics(all_samples_metric, print_table=True)
        result_sample_list.append(s_metric_result)

        # Overlap metric, all batches
        # o_metric_result = (pre, rec, f1, mf1, miou)
        o_metric_result = overlap_metrics(all_overlap_metric, overlap_thresholds, print_threshold=0.2,
                                          print_table=True)
        result_overlap_list.append(o_metric_result)

    if fold == 'all':
        result_sample_array = np.array(result_sample_list)
        result_sample_avg = np.mean(result_sample_array, axis=0)
        result_sample_std = np.std(result_sample_array, axis=0)
        print('\nAverage Samples Metrics:')
        print(f'ACC: {result_sample_avg[0]:.2f} ± {result_sample_std[0]:.2f}, '
              f'PRE: {result_sample_avg[1]:.2f} ± {result_sample_std[1]:.2f}, '
              f'SEN: {result_sample_avg[2]:.2f} ± {result_sample_std[2]:.2f}, '
              f'SPE: {result_sample_avg[3]:.2f} ± {result_sample_std[3]:.2f}, '
              f'F1: {result_sample_avg[4]:.2f} ± {result_sample_std[4]:.2f}')

        results_array = np.array(result_overlap_list)  # shape: (num_ckpts, 5)
        result_avg = np.mean(results_array, axis=0)
        result_std = np.std(results_array, axis=0)

        print('\nAverage Overlap Metrics:')
        print(f'PRE: {result_avg[0]:.2f} ± {result_std[0]:.2f}, '
              f'REC: {result_avg[1]:.2f} ± {result_std[1]:.2f}, '
              f'F1: {result_avg[2]:.2f} ± {result_std[2]:.2f}, '
              f'mF1: {result_avg[3]:.2f} ± {result_std[3]:.2f}, '
              f'mIoU: {result_avg[4]:.2f} ± {result_std[4]:.2f}')


if __name__ == '__main__':
    ckpt_dir = 'weight/MODA_S3Net_fullyfinetune_20250715'
    test_data_dir = r'E:\dataset\MODA\release_test\test'
    test_model(fold='all', ckpt_dir=ckpt_dir, data_dir=test_data_dir)
