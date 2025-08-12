"""
This script is used to generate data for contrastive learning.
Through self supervised learning, we can utilize many data without spindle labels in MASS to enhance the feature
extraction ability of the model backbone.
You need to manually change the location of the paths 'mass_edf_path', 'mass_stage_path' and 'save_dir':
    mass_edf_path: the directory where files '01-01-00XX PSG.edf' etc. are located
    mass_stage_path: the directory where files '01-01-00XX Annotations.edf' etc. are located
    moda_path: the directory where the MODA dataset is located
    save_dir: the directory where the processed data will be saved
"""

import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyedflib

from data.utils import butter_bandpass_filter, resample_data, data_to_epoch
from data.moda_spindle.split_list import split_subject

# Change the following paths to your local paths
mass_edf_path = r'E:\dataset\MASS\edfs'
mass_stage_path = r'E:\dataset\MASS\edfs\annotations\SleepStages'
moda_path = r'E:\dataset\MODA'
save_dir = r'E:\dataset\MASS\release_test'


def find_contiguous_n2(input_arr):

    # find all n2 indices
    indices = np.where(input_arr == '2')[0]

    if len(indices) == 0:
        return []

    # find contiguous indices
    diffs = np.diff(indices)
    break_points = np.where(diffs > 1)[0]
    groups = np.split(indices, break_points + 1)

    # find groups with length >= 2
    result = []
    for group in groups:
        if len(group) >= 2:
            start = group[0]
            end = group[-1]
            result.append([start, end])

    return result


def read_edf(edf_file, stage_file, channel):

    # Read the stage file to get the sleep stages, start times, and durations
    h = pyedflib.EdfReader(stage_file)
    annotations = h.readAnnotations()
    stages = np.array([stage[-1] for stage in annotations[2]])
    stages_start = annotations[0]
    stages_duration = annotations[1]
    stages_duration = np.rint(stages_duration).astype(int)

    # In MASS, there are two staging rules for sleep stages: AASM and R&K.
    # The length of a segment under different rules varies, with 30s and 20s respectively.
    # stages_duration should be all 20 or all 30 seconds,
    # but some files have an incorrect last duration, so they are not considered
    if len(set(stages_duration[:-1])) == 1:
        if stages_duration[0] == 20:
            duration = 20
        else:
            duration = 30
    else:
        print(f'{edf_file} duration not equal')
        return

    # Read the EDF file to get the EEG data
    with pyedflib.EdfReader(edf_file) as f:
        channel_name = f.getSignalLabels()
        if channel == 'C3-A2':
            required_channel = [chn for chn in channel_name if 'C3' in chn][0]
            reference_channel = [chn for chn in channel_name if 'A2' in chn][0]
            required_signal = f.readSignal(channel_name.index(required_channel))
            reference_signal = f.readSignal(channel_name.index(reference_channel))
            required_fs = f.getSampleFrequencies()[channel_name.index(required_channel)]
            reference_fs = f.getSampleFrequencies()[channel_name.index(reference_channel)]
            assert required_fs == reference_fs
            data = required_signal - reference_signal
        else:
            required_channel = [chn for chn in channel_name if 'C3' in chn][0]
            data = f.readSignal(channel_name.index(required_channel))
            required_fs = f.getSampleFrequencies()[channel_name.index(required_channel)]

    # Process the data, including filtering and resampling
    sf = int(required_fs)
    data = butter_bandpass_filter(data, 0.5, 30, sf, order=10)
    data = resample_data(data, sf, 100)
    sf = 100

    # In MASS, there are two staging rules for sleep stages: AASM and R&K.
    # The length of a segment under different rules varies, with 30s and 20s respectively.
    if duration == 30:
        # If the duration is 30 seconds, we can directly extract the n2 segments
        n2_idx = np.where(stages == '2')[0]
        n2_start = stages_start[n2_idx]
        n2_end = stages_start[n2_idx] + stages_duration[n2_idx]

        n2_data = []
        for i in range(len(n2_start)):
            epoch_start = int(n2_start[i] * sf)
            epoch_end = int(n2_end[i] * sf)
            n2_data.append(data[epoch_start:epoch_end])

        n2_data = np.array(n2_data)

    else:
        # In order to process all segments into 30 seconds, for R&K, we find all consecutive n2 -
        # periods and divide them into 30 seconds (if possible, which means there are at least two consecutive n2 (40s))
        contiguous_n2seg_idx = find_contiguous_n2(stages)

        n2_data = []
        for n2seg_idx in contiguous_n2seg_idx:
            n2seg_start_sample = int(stages_start[n2seg_idx[0]] * sf)
            n2seg_end_sample = int((stages_start[n2seg_idx[-1]] + stages_duration[n2seg_idx[-1]]) * sf)
            n2seg_data = data[n2seg_start_sample:n2seg_end_sample]
            n2seg_data = data_to_epoch(n2seg_data, sf, 30, overlap_len_sec=0)
            n2_data.append(n2seg_data)

        n2_data = np.concatenate(n2_data, axis=0)

    return n2_data, sf


if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    edf_files = glob.glob(os.path.join(mass_edf_path, '*.edf'))
    # exclude test set
    edf_files = [x for x in edf_files if os.path.basename(x).split(' ')[0] not in split_subject['test']]

    channel_selected_path = os.path.join(moda_path, 'input', '8_MODA_primChan_180sjt.txt')
    channel_info = pd.read_csv(channel_selected_path, delimiter="\t")
    subjects_and_channel = channel_info.values

    for i, edf_file in tqdm(enumerate(edf_files)):
        subject_id = os.path.basename(edf_file)[:-8]

        channel = subjects_and_channel[np.where(subjects_and_channel[:, 0] == (subject_id + '.edf'))[0], 1]
        if len(channel) == 0:
            channel = 'C3-LE'
        else:
            channel = channel[0]
        print(subject_id, channel)

        save_path = os.path.join(save_dir, f'{subject_id}.npz')
        stage_file = os.path.join(mass_stage_path, f'{subject_id} Base.edf')
        if os.path.exists(save_path):
            print(f'{subject_id} exists')
            continue

        n2_epochs, sf = read_edf(edf_file, stage_file, channel)

        save_dict = {
            'segments': n2_epochs,
            'channel': channel,
            'sampling_rate': sf
        }
        np.savez(save_path, **save_dict)
