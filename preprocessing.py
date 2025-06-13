# Preprocessing of the PPG signal - usually preprocessing is done by filtering
from scipy.signal import savgol_filter
import numpy as np

def preprocess_ppg(ppg_signal, window_length=11, polyorder=3):
    signal_length = len(ppg_signal)

    if window_length % 2 == 0:
        window_length += 1

    if window_length > signal_length:
        window_length = signal_length if signal_length % 2 != 0 else signal_length - 1
        window_length = max(3, window_length)

    if polyorder >= window_length:
        polyorder = window_length - 1
        polyorder = max(1, polyorder)

    if window_length <= polyorder or window_length < 3:
        print("Warning: Could not find valid Savitzky-Golay parameters. Returning original signal.")
        return ppg_signal

    smoothed_ppg = savgol_filter(ppg_signal, window_length, polyorder, mode='interp')
    return smoothed_ppg

def calculate_derivative(signal,time_array):
    gradient_signal = np.diff(signal)/np.diff(time_array)
    gradient_signal = np.append(gradient_signal, gradient_signal[-1])
    return gradient_signal