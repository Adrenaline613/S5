import pyedflib
import torch
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from lxml import etree
import re


def load_edf(path, ch):
    f = pyedflib.EdfReader(path)

    signal_labels = f.getSignalLabels()
    idx_chan = signal_labels.index(ch)

    data = f.readSignal(idx_chan)
    sf = f.getSampleFrequency(idx_chan)

    unit = f.getPhysicalDimension(idx_chan)
    if unit != 'uV':
        print('unit:', unit)
        if unit == 'mV':
            data = data * 1000

    times = data.shape[0] / sf

    return data, sf, times


def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    sos_high = butter(order, lowcut, btype='hp', fs=sampling_rate, output='sos')
    sos_low = butter(order, highcut, btype='lp', fs=sampling_rate, output='sos')

    filtered_data = sosfiltfilt(sos_low, sosfiltfilt(sos_high, data, padlen=3 * order), padlen=3 * order)

    return filtered_data


def resample_data(data, sampling_rate, resampling_rate):
    sampling_rate = int(sampling_rate)
    resampling_rate = int(resampling_rate)

    if sampling_rate == resampling_rate:
        return data

    gcd = np.gcd(sampling_rate, resampling_rate)
    up = resampling_rate // gcd
    down = sampling_rate // gcd

    resampled_data = resample_poly(data, up, down)

    return resampled_data


def data_to_epoch(data, sf, epoch_len_sec=30, overlap_len_sec=1, padding_mode='constant'):
    """
    Segment time series data into multiple epochs with optional overlap before and after

    Parameters:
        data (np.ndarray): One-dimensional time series data
        sf (int): Sampling frequency
        epoch_len_sec (int): Length of each segment in seconds
        overlap_len_sec (int): Overlap length in seconds
        padding_mode (str): Padding method, supports all modes supported by numpy.pad, e.g. 'constant' (zero padding),
        'edge' (nearest value padding)

    Returns:
        np.ndarray: shape = (num_epochs, epoch_points)
    """
    # Convert to sample points
    epoch_len = int(epoch_len_sec * sf)
    overlap_len = int(overlap_len_sec * sf)

    # Total data length
    total_data_len = data.shape[0]

    # Calculate how many complete main segments can be extracted (excluding the final incomplete part)
    num_full_epochs = total_data_len // epoch_len

    # Construct all epochs
    epochs = []

    for i in range(num_full_epochs):
        # Main segment start position
        start = i * epoch_len
        end = start + epoch_len

        # Context start and end positions
        context_start = start - overlap_len
        context_end = end + overlap_len

        # Extract the area around the current main segment (may exceed the original data range)
        if context_start < 0 or context_end > total_data_len:
            # Need padding
            left_pad = max(-context_start, 0)
            right_pad = max(context_end - total_data_len, 0)

            segment = data[max(context_start, 0):min(context_end, total_data_len)]
            padded_segment = np.pad(segment, (left_pad, right_pad), mode=padding_mode)
            epochs.append(padded_segment)
        else:
            epochs.append(data[context_start:context_end])

    return np.array(epochs)


def read_xml(xml_file_path):
    # Parse XML
    root = etree.parse(xml_file_path)

    # Use XPath to find all related ScoredEvent
    events = root.xpath("//ScoredEvent[contains(Name, 'Hypopnea') or contains(Name, 'Apnea')]")

    sahs_count = 0
    # Print results
    for event in events:
        duration = event.find('Duration').text
        if float(duration) >= 10:
            sahs_count += 1

    with open(xml_file_path, 'r') as f:
        xml = f.read()

    pattern = re.compile(r'<SleepStage>(.*?)</SleepStage>')
    sleep_stages = pattern.findall(xml)

    epoch_length_pattern = re.compile(r'<EpochLength>(.*?)</EpochLength>')
    epoch_length = int(epoch_length_pattern.findall(xml)[0])

    return sleep_stages, epoch_length, sahs_count
