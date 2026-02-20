import numpy as np
import pandas as pd
from mne.filter import filter_data
from scipy import signal
from scipy.fftpack import next_fast_len


def spindle_feature(
        data,
        label,
        sf,
        freq_broad=(1, 30),
        freq_sp=(11, 16)
):
    """
    Extract spindle features from a single EEG channel.

    Parameters
    ----------
    data: array_like
        1D array of EEG data.
    label: array_like
        1D array of sleep stage labels.
    sf: float
        Sampling frequency of the data.
    freq_broad: list
        Broadband frequency range for spindle detection.
    freq_sp: list
        Sigma frequency range for spindle detection.
    duration: list
        Duration range for spindle detection.

    Returns
    -------
    df: pd.DataFrame
        A dataframe containing the spindle features.

        notes:
        1. The start, peak and end are expressed in seconds.
        2. The amplitude is expressed in uV.
        3. The median frequency and absolute power of each spindle is computed using an Hilbert transform.
        4. The number of oscillations was based on counting peaks and troughs
        5. The symmetry was based on the location of the maximum peak-to-peak change relative to the
        start (0.0) and end (1.0) of the spindle interval.
    """

    # Convert the label to a numpy array
    where_sp = np.where(label == 1)[0]

    if len(where_sp) == 0:
        return pd.DataFrame({
            "Start": [],
            "Peak": [],
            "End": [],
            "Duration": [],
            "Amplitude": [],
            "Frequency": [],
            "Oscillations": [],
            "Symmetry": []
        })

    sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
    idx_start_end = np.array([[k[0], k[-1]] for k in sp]) / sf
    sp_start, sp_end = idx_start_end.T
    sp_dur = sp_end - sp_start

    # Initialize the output arrays
    sp_amp = np.zeros(len(sp_dur))
    sp_freq = np.zeros(len(sp_dur))
    sp_osc = np.zeros(len(sp_dur))
    sp_sym = np.zeros(len(sp_dur))
    sp_abs = np.zeros(len(sp_dur))
    sp_pro = np.zeros(len(sp_dur))

    # Find events with bad duration
    good_dur = np.logical_and(sp_dur > 0.1, sp_dur < 5.0)

    # Filter the data
    if freq_broad is not None:
        data_broad = filter_data(data, sf, freq_broad[0], freq_broad[1])
    else:
        data_broad = data
    data_sigma = filter_data(
        data,
        sf,
        freq_sp[0],
        freq_sp[1],
    )

    # Hilbert power (to define the instantaneous frequency / power)
    n_samples = data.shape[0]
    n_fast = next_fast_len(n_samples)
    analytic = signal.hilbert(data_sigma, N=n_fast)[:n_samples]
    inst_phase = np.angle(analytic)
    inst_pow = np.square(np.abs(analytic))
    inst_freq = sf / (2 * np.pi) * np.diff(inst_phase, axis=-1)

    for j in np.arange(len(sp_dur))[good_dur]:

        # detrend the signal to avoid wrong PTP amplitude
        sp_det = signal.detrend(data_broad[sp[j]], type='linear')

        # Peak-to-peak amplitude
        sp_amp[j] = np.ptp(sp_det)

        # Hilbert-based instantaneous properties
        sp_inst_freq = inst_freq[sp[j]]
        sp_inst_pow = inst_pow[sp[j]]
        sp_abs[j] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
        sp_freq[j] = np.median(sp_inst_freq[sp_inst_freq > 0])

        # Number of oscillations (number of peaks separated by at least 60 ms)
        # --> 60 ms because 1000 ms / 16 Hz = 62.5 m, in other words, at 16 Hz,
        # peaks are separated by 62.5 ms. At 11 Hz peaks are separated by 90 ms
        distance = 60 * sf / 1000
        peaks, peaks_params = signal.find_peaks(
            sp_det, distance=distance, prominence=(None, None)
        )
        sp_osc[j] = len(peaks)

        # Peak location & symmetry index
        pk = peaks[peaks_params["prominences"].argmax()]
        sp_pro[j] = sp_start[j] + pk / sf
        sp_sym[j] = pk / sp_det.size

    sp_feature = {
            "Start": sp_start,
            "Peak": sp_pro,
            "End": sp_end,
            "Duration": sp_dur,
            "Amplitude": sp_amp,
            "Frequency": sp_freq,
            "Oscillations": sp_osc,
            "Symmetry": sp_sym,
        }

    df = pd.DataFrame(sp_feature)[good_dur]
    return df


def get_centered_indices(data, idx, npts_before, npts_after):
    # Safety check
    assert isinstance(npts_before, (int, float))
    assert isinstance(npts_after, (int, float))
    assert float(npts_before).is_integer()
    assert float(npts_after).is_integer()
    npts_before = int(npts_before)
    npts_after = int(npts_after)
    data = np.asarray(data)
    idx = np.asarray(idx, dtype="int")
    assert idx.ndim == 1, "idx must be 1D."
    assert data.ndim == 1, "data must be 1D."

    def rng(x):
        """Create a range before and after a given value."""
        return np.arange(x[0] - npts_before, x[0] + npts_after + 1, dtype="int")

    idx_ep = np.apply_along_axis(rng, 1, idx[..., np.newaxis])
    # We drop the events for which the indices exceed data
    idx_ep = np.ma.mask_rows(np.ma.masked_outside(idx_ep, 0, data.shape[0]))
    # Indices of non-masked (valid) epochs in idx
    idx_ep_nomask = np.unique(idx_ep.nonzero()[0])
    idx_ep = np.ma.compress_rows(idx_ep)
    return idx_ep, idx_ep_nomask


def get_sync_events(
    data, event, sf, center, time_before, time_after, filt=(None, None), as_dataframe=True
):
    assert time_before >= 0
    assert time_after >= 0
    bef = int(sf * time_before)
    aft = int(sf * time_after)
    time = np.arange(-bef, aft + 1, dtype="int") / sf

    if any(filt):
        data = filter_data(
            data, sf, l_freq=filt[0], h_freq=filt[1]
        )

    output = []

    for i in event["SegmentsIdx"].unique():
        ev_seg = event[event["SegmentsIdx"] == i].copy()
        ev_seg["Event"] = np.arange(ev_seg.shape[0])
        peaks = (ev_seg[center] * sf).astype(int).to_numpy()
        # Get centered indices
        idx, idx_valid = get_centered_indices(data[i, :], peaks, bef, aft)
        # If no good epochs are returned raise a warning
        if len(idx_valid) == 0:
            print("No valid epochs found.")
            continue

        # Get data at indices and time vector
        amps = data[i, idx]

        if not as_dataframe:
            # Output is a list (n_channels) of numpy arrays (n_events, n_times)
            output.append(amps)
            continue

        # Convert to long-format dataframe
        df_chan = pd.DataFrame(amps.T)
        df_chan["Time"] = time
        # Convert to long-format
        df_chan = df_chan.melt(id_vars="Time", var_name="Event", value_name="Amplitude")

        # Append channel name
        df_chan["Channel"] = ev_seg["Channel"].iloc[0]
        df_chan["SegmentsIdx"] = i
        # Append to master dataframe
        output.append(df_chan)

    if as_dataframe:
        output = pd.concat(output, ignore_index=True)

    return output
