import torch
import random
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.ndimage.interpolation import shift


class SpindleBandScale:
    """
    Scales the amplitude of slow wave, theta, and sigma bands in the input data.
    This transformation can either enhance or weaken the selected bands.
    The probability of applying this transformation is determined by the 'p' parameter.
    """

    def __init__(self, slow_wave=(0.5, 2), theta=(4, 8), sigma=(11, 16), scale_range=(0.5, 2), sf=100.0, p=0.8):
        self.slow_wave = slow_wave
        self.theta = theta
        self.sigma = sigma
        self.scale_range = scale_range
        self.sf = sf
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            slow_wave_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            x = adjust_amplitude(x, self.sf, self.slow_wave, slow_wave_factor)

        if torch.rand(1) < self.p:
            theta_factor = random.uniform(self.scale_range[0], 1)
            x = adjust_amplitude(x, self.sf, self.theta, theta_factor)

        if torch.rand(1) < self.p:
            sigma_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            x = adjust_amplitude(x, self.sf, self.sigma, sigma_factor)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AmplitudeScale:
    """
    Scales the amplitude of the input data.
    This transformation can either enhance or weaken the overall amplitude.
    The probability of applying this transformation is determined by the 'p' parameter.
    """

    def __init__(self, scale_range=(0.5, 2.0), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            return x * scale
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TimeShift:
    """
    Shifts the input data in time.
    This transformation can move the data forward or backward in time.
    The probability of applying this transformation is determined by the 'p' parameter.
    """

    def __init__(self, shift_range=(-300, 300), cval=0.0, p=0.5):
        self.shift_range = shift_range
        self.cval = cval
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            t_shift = random.randint(self.shift_range[0], self.shift_range[1])
            if len(x.shape) == 2:
                x = x[0]
            x = shift(input=x, shift=t_shift, mode='constant', cval=self.cval)
            x = np.expand_dims(x, axis=0)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ZeroMask:
    """
    Masks a random segment of the input data with zeros.
    This transformation can weaken the signal by introducing zeros.
    The probability of applying this transformation is determined by the 'p' parameter.
    """

    def __init__(self, mask_range=(0, 300), p=0.5):
        self.mask_range = mask_range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            mask_len = random.randint(self.mask_range[0], self.mask_range[1])
            random_pos = random.randint(0, x.shape[1] - mask_len)
            mask = np.concatenate(
                [np.ones((1, random_pos)), np.zeros((1, mask_len)), np.ones((1, x.shape[1] - mask_len - random_pos))],
                axis=1)
            return x * mask
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AdditiveNoise:
    """
    Adds Gaussian noise to the input data.
    This transformation can weaken the signal by introducing random noise.
    The probability of applying this transformation is determined by the 'p' parameter.
    """

    def __init__(self, n_range=(0.0, 0.2), p=0.5):
        self.n_range = n_range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            sigma = random.uniform(self.n_range[0], self.n_range[1])
            return x + np.random.normal(0, sigma, x.shape)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class BandStop:
    """
    Applies a band-stop filter to the input data.
    This transformation can weaken the selected frequency bands by attenuating them.
    The probability of applying this transformation is determined by the 'p' parameter.
    """

    def __init__(self, band_range=(0.5, 30.0), band_stop_width=2.0, sf=100.0, p=0.5):
        self.band_range = band_range
        self.band_stop_width = band_stop_width
        self.sf = sf
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            low_freq = random.uniform(self.band_range[0], self.band_range[1])
            center_freq = low_freq + self.band_stop_width / 2.0
            b, a = signal.iirnotch(center_freq, center_freq / self.band_stop_width, fs=self.sf)
            x = signal.lfilter(b, a, x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose:
    """
    Initialize the Compose class with a list of transformations and a mode.

    Args:
        transforms (list): List of transformation functions.
        mode (str): Mode of transformation application. Options are 'random', 'full', and 'shuffle'.
    """

    def __init__(self, transforms, mode='full'):
        self.transforms = transforms
        self.mode = mode

    def __call__(self, x):
        """
        Apply transformations to the input data based on the specified mode.

        Args:
            x (any): Input data to be transformed.

        Returns:
            Transformed data.
        """
        if self.mode == 'random':
            index = random.randint(0, len(self.transforms) - 1)
            x = self.transforms[index](x)
        elif self.mode == 'full':
            for t in self.transforms:
                x = t(x)
        elif self.mode == 'shuffle':
            transforms = np.random.choice(self.transforms, len(self.transforms), replace=False)
            for t in transforms:
                x = t(x)
        else:
            raise NotImplementedError
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TwoTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def bandpass_filter(data, lowcut, highcut, sf, order=4):
    # Bandpass filter the data using a Butterworth filter.
    nyq = 0.5 * sf
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def adjust_amplitude(data, sf, band, factor):
    # Adjust the amplitude of the specified frequency band in the data.
    filtered = bandpass_filter(data, band[0], band[1], sf)
    adjusted = filtered * factor
    return data - filtered + adjusted
