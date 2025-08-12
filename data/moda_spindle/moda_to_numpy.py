"""
This script is used to convert the MODA dataset to numpy format.
Please run this script to generate data before training and testing.
You need to manually change the location of the paths 'mass_edf_path', 'moda_path', and 'save_dir'
    'mass_edf_path': the directory where files '01-01-00XX PSG.edf' etc. are located
    'moda_path': the directory where the MODA dataset is located
    'save_dir': the directory where the numpy files will be saved
    'fixed': If you want to fully reproduce the results and make fair comparisons, please use True
"""

import os
import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm
from scipy.stats import zscore

from data.utils import butter_bandpass_filter, resample_data
from data.moda_spindle.split_list import split_subject


# Change the following paths to your local paths
mass_edf_path = r'E:\dataset\MASS\edfs'
moda_path = r'E:\dataset\MODA'
save_dir = r'E:\dataset\MODA\release_test'
fixed = True    # If you want to fully reproduce the results and make fair comparisons, please use True


def get_subjects_and_channel(moda_base_path):
    """
    Get the subject id, channel, and phase of each subject in the MODA dataset.

    Args:
        moda_base_path (str): The base path of the MODA dataset.

    Returns:
        list: A list of dictionaries containing subject id, channel, and phase.
    """

    # Define paths to the necessary files
    channel_selected_path = os.path.join(moda_base_path, 'input', '8_MODA_primChan_180sjt.txt')
    phase_1_path = os.path.join(moda_base_path, 'input', '6_segListSrcDataLoc_p1.txt')
    phase_2_path = os.path.join(moda_base_path, 'input', '7_segListSrcDataLoc_p2.txt')

    # Read channel information
    channel_info = pd.read_csv(channel_selected_path, delimiter="\t")
    subjects_and_channel = channel_info.values

    # Read phase information
    phase_1_subjects = pd.read_csv(phase_1_path, delimiter="\t")
    phase_2_subjects = pd.read_csv(phase_2_path, delimiter="\t")
    phase_1_subjects = np.unique(phase_1_subjects.subjectID.values)
    phase_2_subjects = np.unique(phase_2_subjects.subjectID.values)

    # Combine subject id, channel, and phase information
    subjects_channel_phase = [{'id': subjects_and_channel[i][0][:-4], 'ch': subjects_and_channel[i][1],
                             'p': 1 if subjects_and_channel[i][0][:-4] in phase_1_subjects else 2}
                            for i in range(len(subjects_and_channel))]

    return subjects_channel_phase


def get_data_and_label(mass_base_path, moda_base_path, subject_id, channel, seg_duration=115, resample_rate=100, is_filter=True, is_norm=True):
    """
    Get the data and labels for a given subject and channel.

    Args:
        mass_base_path (str): The base path of the MASS dataset.
        moda_base_path (str): The base path of the MODA dataset.
        subject_id (str): The subject id.
        channel (str): The channel name.
        seg_duration (int, optional): The duration of each segment in seconds. Default is 115.
        resample_rate (int, optional): The resampling rate. Default is 100.
        is_filter (bool, optional): Whether to apply a bandpass filter. Default is True.
        is_norm (bool, optional): Whether to normalize the data. Default is True.

    Returns:
        tuple: A tuple containing the segments and labels.
    """

    # Define paths to the EDF and annotation files
    edf_path = os.path.join(mass_base_path, subject_id + ' PSG.edf')
    ann_path = os.path.join(moda_base_path, 'output', 'exp', 'annotFiles', subject_id + '_MODA_GS.txt')

    # Read the EDF file and extract the required channel data
    with pyedflib.EdfReader(edf_path) as f:
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
            required_channel = [chn for chn in channel_name if channel in chn][0]
            data = f.readSignal(channel_name.index(required_channel))
            required_fs = f.getSampleFrequencies()[channel_name.index(required_channel)]

    sf = int(required_fs)

    # Read the annotation file
    annot = pd.read_csv(ann_path, delimiter="\t")

    # Extract segment start times
    segments_info = annot[annot.eventName == "segmentViewed"]
    segments_start = np.sort(segments_info.startSec.values)

    # Extract spindle start and end times
    spindles_info = annot[annot.eventName == "spindle"]
    spindles_start = spindles_info.startSec.values
    spindles_end = spindles_info.durationSec.values + spindles_start

    # Obtain data segments and perform necessary preprocessing
    segments = []
    labels = []
    for seg_start in segments_start:
        start_sample = int(seg_start * sf)
        end_sample = int((seg_start + seg_duration) * sf)
        segment = data[start_sample:end_sample]
        if is_filter:
            segment = butter_bandpass_filter(segment, 0.3, 30, sf, 10)
        segment = resample_data(segment, sf, resample_rate)
        if is_norm:
            segment = zscore(segment)
        segments.append(segment)

        label = np.zeros_like(segment, dtype=np.int8)
        for i in range(len(spindles_start)):
            if spindles_start[i] < seg_start + seg_duration and spindles_end[i] > seg_start:
                spindles_start_sample = int((spindles_start[i] - seg_start) * resample_rate)
                spindles_end_sample = int((spindles_end[i] - seg_start) * resample_rate)
                label[spindles_start_sample:spindles_end_sample] = 1
        labels.append(label)

    return segments, labels


if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    subjects_channel_phase_dict = get_subjects_and_channel(moda_path)

    for i in tqdm(range(len(subjects_channel_phase_dict))):
        # get subject id
        save_name = subjects_channel_phase_dict[i]['id'] + '_p' + str(subjects_channel_phase_dict[i]['p']) + '.npz'

        # check if the file already exists
        if os.path.exists(os.path.join(save_dir, save_name)):
            print(f'{save_name} already exists, skip')
            continue

        # get data and label
        segments, labels = get_data_and_label(
            mass_edf_path,
            moda_path,
            subjects_channel_phase_dict[i]['id'],
            subjects_channel_phase_dict[i]['ch'],
            seg_duration=115,
            resample_rate=100,
            is_filter=True,
            is_norm=False
        )

        # save data
        save_dict = {
            'subject_id': subjects_channel_phase_dict[i]['id'],
            'channel': subjects_channel_phase_dict[i]['ch'],
            'sampling_rate': 100,
            'phase': subjects_channel_phase_dict[i]['p'],
            'segments': segments,
            'labels': labels
        }

        np.savez(os.path.join(save_dir, save_name), **save_dict)

    # If fixed data partitioning is used, fix the dataset according to the dictionary split_subject
    # First, create subfolders based on key
    if fixed:
        for key in split_subject.keys():
            if not os.path.exists(os.path.join(save_dir, key)):
                os.makedirs(os.path.join(save_dir, key))

        # Move the processed dataset to the corresponding folder based on the key value pairs in split_subject
        for i in os.listdir(save_dir):
            if i.endswith('.npz'):
                for key, value in split_subject.items():
                    if i.split('_')[0] in value:
                        os.rename(os.path.join(save_dir, i), os.path.join(save_dir, key, i))
                        break
