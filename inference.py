import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from PyQt5.QtWidgets import QApplication
from plot.qt_visualizer import EpochViewer

from data.utils import butter_bandpass_filter, resample_data, load_edf, data_to_epoch, read_xml
from model.metric import *
from model.postprocessing import process_events
from model.net import S3Net


def load_eeg(file_path, annot_path, channel='EEG(sec)', r_sf=100, is_filter=True, n2_only = True):
    # Load EEG data
    eeg, sf, times = load_edf(file_path, ch=channel)

    # Filter
    if is_filter:
        eeg = butter_bandpass_filter(eeg, lowcut=1, highcut=30, sampling_rate=sf, order=10)

    # Resample
    eeg = resample_data(eeg, sf, r_sf)

    # Seg
    epochs = data_to_epoch(eeg, r_sf, epoch_len_sec=30, overlap_len_sec=0)

    # Read Xml, get sleep stages, epoch length and num of osa(hypopnea)
    sleep_stages, epoch_length, sahs_count = read_xml(annot_path)

    assert epoch_length == 30, f"Expected epoch length of 30 seconds, but got {epoch_length} seconds."
    assert len(sleep_stages) == len(epochs), f"Number of sleep stages ({len(sleep_stages)}) does not match."

    sleep_stages = np.array(sleep_stages)
    if n2_only:
        valid_indices = np.where(sleep_stages == '2')[0]
        sleep_stages = sleep_stages[valid_indices]
        epochs = epochs[valid_indices]

    return epochs


class InferDataset(Dataset):
    def __init__(self, x):

        x = x.astype(np.float32)

        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(1, 0, 2)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        print('ok')

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


def main():

    data_path = r'edf4test/shhs1-eeg-only.edf'
    annot_path = r'edf4test/shhs1-eeg-only-profusion.xml'
    epochs = load_eeg(file_path=data_path, annot_path=annot_path, channel='EEG(sec)', r_sf=100, is_filter=True, n2_only=True)
    dataset = InferDataset(epochs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)

    ckpt_path = r'weight/MODA_S3Net_fullyfinetune_20250715/fold1_best.ckpt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = S3Net(in_channels=1, num_classes=2, training_mode='scratch')
    model = model.to(device)
    model_checkpoint = torch.load(ckpt_path)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)

            inputs = inputs.cpu().numpy()
            preds = preds.cpu().numpy()
            for idx, pred in enumerate(preds):
                preds[idx] = process_events(pred, 100, 0.5, 3, 0.3)
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)

    return epochs, all_preds


if __name__ == '__main__':
    all_epochs, all_preds = main()

    app = QApplication(sys.argv)
    viewer = EpochViewer(all_epochs, all_preds*50, (-200, 200))
    viewer.show()
    sys.exit(app.exec_())

