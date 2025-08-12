import numpy as np
import torch


def process_events(events, fs, min_duration=0.3, max_duration=3, sep=0.3):
    """
    Process event annotations by filtering out short events, merging close events, and capping long events.

    Args:
        events (numpy.ndarray or torch.Tensor): Event annotations where 1 indicates an event and 0 indicates no event.
        fs (int): Sampling frequency of the events.
        min_duration (float): Minimum duration of an event in seconds. Default is 0.3 seconds.
        max_duration (float): Maximum duration of an event in seconds. Default is 3 seconds.
        sep (float): Maximum separation between events to be merged in seconds. Default is 0.3 seconds.

    Returns:
        numpy.ndarray: Processed event annotations.
    """

    if isinstance(events, torch.Tensor):
        events = events.cpu().numpy()

    # Convert durations from seconds to samples
    min_samples = int(min_duration * fs)
    max_samples = int(max_duration * fs)
    sep_samples = int(sep * fs)

    # Find the start and end indices of events
    event_starts = np.where(np.diff(events, prepend=0) == 1)[0]
    event_ends = np.where(np.diff(events, append=0) == -1)[0]

    # If there are no events, return the original array
    if len(event_starts) == 0 or len(event_ends) == 0:
        return events

    # Merge events closer than sep_duration
    merged_starts = []
    merged_ends = []
    current_start = event_starts[0]
    current_end = event_ends[0]

    for start, end in zip(event_starts[1:], event_ends[1:]):
        if start - current_end <= sep_samples:
            current_end = end
        else:
            merged_starts.append(current_start)
            merged_ends.append(current_end)
            current_start = start
            current_end = end

    merged_starts.append(current_start)
    merged_starts = np.array(merged_starts)
    merged_ends.append(current_end)
    merged_ends = np.array(merged_ends)

    # Filter out events shorter than min_duration
    valid_events = (merged_ends - merged_starts) >= min_samples
    merged_shortfil_starts = merged_starts[valid_events]
    merged_shortfil_ends = merged_ends[valid_events]

    # Process events bigger than max_duration
    new_starts = []
    new_ends = []
    for start, end in zip(merged_shortfil_starts, merged_shortfil_ends):
        duration = end - start
        if duration > 2 * max_samples:
            continue  # Remove events longer than 2 * max_duration
        elif duration > max_samples:
            center = (start + end) // 2
            start = center - max_samples // 2
            end = start + max_samples
        new_starts.append(start)
        new_ends.append(end)

    # Create the processed events array
    processed_events = np.zeros_like(events)
    for start, end in zip(new_starts, new_ends):
        processed_events[start:end] = 1

    return processed_events
