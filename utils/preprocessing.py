from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, iirnotch, sosfiltfilt, iirfilter
from scipy.ndimage.filters import maximum_filter1d
import scipy.signal as signal
import warnings
import math

import numpy as np
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)

from scipy.signal import sosfilt
from scipy.signal import zpk2sos


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_notch_filter(x, fscut, fs, Q=30.0):
    w0 = fscut / (fs / 2)  # Normalized Frequency
    # Design notch filter
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, x)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def iir_bandpass(data, freqmin, freqmax, df, corners=4, zerophase=True):
    """
    :copyright:
    The ObsPy Development Team (devs@obspy.org)

    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def highpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def eclipse_distance(a, b):
    return math.sqrt(math.pow((a - b), 2))


def agglomerative_clustering(labels, fs):
    positions = np.where(labels == 1)[0]
    groups = []
    groups_len = []
    if len(positions) > 0:
        groups_temp = [positions[0]]
        for index in range(1, len(positions)):
            if eclipse_distance(positions[index], groups_temp[-1]) > 0.080 * fs:
                beat_position = int(np.mean(groups_temp))
                groups.append(beat_position)
                groups_len.append(len(groups_temp))

                groups_temp.clear()

            groups_temp.append(positions[index])

        if len(groups_temp) > 0:
            groups.append(int(np.mean(groups_temp)))
            groups_len.append(len(groups_temp))
        groups = np.asarray(groups)
        groups_len = np.asarray(groups_len)

    return groups, groups_len


vaidyanathan = ([0.045799334111000, 0.250184129505000, 0.572797793211000, 0.635601059872000, 0.201612161775000,
                 -0.263494802488000, -0.194450471766000, 0.135084227129000, 0.131971661417000, -0.083928884366000,
                 -0.077709750902000, 0.055892523691000, 0.038742619293000, -0.035470398607000, -0.014853448005000,
                 0.019687215010000, 0.003153847056000, -0.008839103409000, 0.000708137504000, 0.002843834547000,
                 -0.000944897136000, -0.000453956620000, 0.000343631905000, -0.000062906118000],
                [0.000062906118000, 0.000343631905000, 0.000453956620000, -0.000944897136000, -0.002843834547000,
                 0.000708137504000, 0.008839103409000, 0.003153847056000, -0.019687215010000, -0.014853448005000,
                 0.035470398607000, 0.038742619293000, -0.055892523691000, -0.077709750902000, 0.083928884366000,
                 0.131971661417000, -0.135084227129000, -0.194450471766000, 0.263494802488000, 0.201612161775000,
                 -0.635601059872000, 0.572797793211000, -0.250184129505000, 0.045799334111000],
                [-0.000062906118000, 0.000343631905000, -0.000453956620000, -0.000944897136000, 0.002843834547000,
                 0.000708137504000, -0.008839103409000, 0.003153847056000, 0.019687215010000, -0.014853448005000,
                 -0.035470398607000, 0.038742619293000, 0.055892523691000, -0.077709750902000, -0.083928884366000,
                 0.131971661417000, 0.135084227129000, -0.194450471766000, -0.263494802488000, 0.201612161775000,
                 0.635601059872000, 0.572797793211000, 0.250184129505000, 0.045799334111000],
                [0.045799334111000, -0.250184129505000, 0.572797793211000, -0.635601059872000, 0.201612161775000,
                 0.263494802488000, -0.194450471766000, -0.135084227129000, 0.131971661417000, 0.083928884366000,
                 -0.077709750902000, -0.055892523691000, 0.038742619293000, 0.035470398607000, -0.014853448005000,
                 -0.019687215010000, 0.003153847056000, 0.008839103409000, 0.000708137504000, -0.002843834547000,
                 -0.000944897136000, 0.000453956620000, 0.000343631905000, 0.000062906118000])


def beat_annotations(annotation):
    """ Get rid of non-beat markers """
    good = ['N', 'L', 'R', 'A', 'V', 'a', 'F', 'j', 'f', 'E', 'J', 'e', 'Q', 'S']
    ids = np.in1d(annotation.symbol, good)
    samples = annotation.sample[ids]
    symbols = np.asarray(annotation.symbol)[ids]
    return samples, symbols
