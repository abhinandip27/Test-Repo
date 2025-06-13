import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# --- 1. Signal Simulation and Preprocessing Functions ---

def simulate_ppg_signal(fs, duration_sec, heart_rate_bpm):
    """Generates a synthetic PPG signal with typical morphology."""
    num_samples = int(fs * duration_sec)
    time = np.linspace(0, duration_sec, num_samples, endpoint=False)
    heart_rate_hz = heart_rate_bpm / 60
    
    # Base sinusoidal component for systolic peak
    ppg_signal_base = np.sin(2 * np.pi * heart_rate_hz * time)
    
    # Add a dicrotic notch (higher frequency component slightly delayed)
    dicrotic_component = -0.25 * np.sin(2 * np.pi * 5 * heart_rate_hz * (time - 0.1)) 
    
    # Combine to simulate a more realistic PPG shape
    ppg_signal_dummy = ppg_signal_base + dicrotic_component 
    
    # Apply some damping to make the decay more realistic within each beat
    time_within_beat = np.fmod(time, (1.0 / heart_rate_hz)) 
    ppg_signal_dummy = ppg_signal_dummy * np.exp(-5 * time_within_beat) 
    
    # Add a slow baseline wander and scale the signal to a typical range (0-1)
    ppg_signal_dummy = ppg_signal_dummy + np.sin(2 * np.pi * 0.1 * time) * 0.1 
    ppg_signal_dummy = (ppg_signal_dummy - np.min(ppg_signal_dummy)) / (np.max(ppg_signal_dummy) - np.min(ppg_signal_dummy))
    ppg_signal_dummy = ppg_signal_dummy * 0.8 + 0.2 # Scale to 0.2 to 1.0 range
    
    # Add a small amount of Gaussian noise
    ppg_signal_dummy += np.random.normal(0, 0.02, num_samples)
    
    return time, ppg_signal_dummy

def preprocess_ppg(signal, fs, lowcut=0.5, highcut=8.0, order=4):
    """Applies a Butterworth bandpass filter to the PPG signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def calculate_derivative(signal, time):
    """Calculates the numerical derivative of a signal with respect to time."""
    return np.gradient(signal, time)

# --- 2. APG Fiducial Point Detection for a Single Beat ---

def find_apg_fiducials_for_single_segment(apg_signal, beat_start_idx, beat_end_idx, fs):
    """
    Identifies 'a', 'b', 'c', 'd', 'e' waves within a single APG beat segment.
    Returns the global indices of these points.
    """
    if beat_start_idx >= len(apg_signal) or beat_end_idx > len(apg_signal) or beat_start_idx >= beat_end_idx:
        return None, None, None, None, None

    apg_beat_segment = apg_signal[beat_start_idx:beat_end_idx]
    
    a_loc, b_loc, c_loc, d_loc, e_loc = None, None, None, None, None

    if len(apg_beat_segment) < 5: 
        return None, None, None, None, None

    apg_segment_range = np.max(apg_beat_segment) - np.min(apg_beat_segment)
    if apg_segment_range <= 0.0001: # Handle very flat segments
        return None, None, None, None, None

    beat_duration_sec = (beat_end_idx - beat_start_idx) / fs
    if beat_duration_sec == 0: 
        return None, None, None, None, None

    # APG specific parameters: smaller distance, very low prominence to catch all local extrema
    apg_min_peak_distance = int(fs * 0.02) # 20ms distance
    apg_prominence_for_all_extrema = 0.0001 # A small absolute value

    pos_peaks_local_idx, _ = find_peaks(apg_beat_segment, distance=apg_min_peak_distance, prominence=apg_prominence_for_all_extrema)
    neg_peaks_local_idx, _ = find_peaks(-apg_beat_segment, distance=apg_min_peak_distance, prominence=apg_prominence_for_all_extrema)

    # Convert local indices to global indices and get corresponding APG values
    all_pos_peaks_global_info = [(idx + beat_start_idx, apg_beat_segment[idx]) for idx in pos_peaks_local_idx]
    all_neg_peaks_global_info = [(idx + beat_start_idx, apg_beat_segment[idx]) for idx in neg_peaks_local_idx]
    
    # --- Robust APG Fiducial Point Identification Logic ---
    
    # 1. Identify 'a' wave: The highest positive peak in the early part of the beat.
    # Search window for 'a': 0% to 30% of the beat duration from beat start.
    a_search_start_time = beat_start_idx / fs
    a_search_end_time = a_search_start_time + (0.3 * beat_duration_sec) 
    a_candidates = [p for p in all_pos_peaks_global_info if a_search_start_time <= p[0] / fs <= a_search_end_time]
    if a_candidates:
        a_loc = max(a_candidates, key=lambda x: x[1])[0] 

    if a_loc is None: 
        return None, None, None, None, None

    # 2. Identify 'b' wave: The most negative peak *after* 'a'. 
    # Search window for 'b': from a_loc time + 5% of beat duration to a_loc time + 40% of beat duration.
    b_search_start_time = (a_loc / fs) + (0.05 * beat_duration_sec) 
    b_search_end_time = (a_loc / fs) + (0.4 * beat_duration_sec) 
    b_candidates = [p for p in all_neg_peaks_global_info if b_search_start_time <= p[0] / fs <= b_search_end_time]
    if b_candidates:
        b_loc = min(b_candidates, key=lambda x: x[1])[0] 

    current_anchor_time_sec = (b_loc / fs) if b_loc is not None else (a_loc / fs)

    if current_anchor_time_sec is not None:
        # --- Find 'c' wave ---
        c_search_start_time = current_anchor_time_sec + (0.05 * beat_duration_sec) 
        c_search_end_time = (a_loc / fs) + (0.5 * beat_duration_sec)
        c_candidates = [p for p in all_pos_peaks_global_info if c_search_start_time <= p[0] / fs <= c_search_end_time]
        if c_candidates:
            c_loc = sorted(c_candidates, key=lambda x: (x[0], -x[1]))[0][0] 
            current_anchor_time_sec = (c_loc / fs)
        else:
            c_loc = None 

        # --- Find 'd' wave ---
        d_search_start_time = (current_anchor_time_sec + (0.05 * beat_duration_sec)) if c_loc is not None else ((b_loc / fs if b_loc is not None else a_loc / fs) + (0.1 * beat_duration_sec))
        d_search_end_time = (a_loc / fs) + (0.6 * beat_duration_sec)
        d_candidates = [p for p in all_neg_peaks_global_info if d_search_start_time <= p[0] / fs <= d_search_end_time]
        if d_candidates:
            d_loc = sorted(d_candidates, key=lambda x: (x[0], x[1]))[0][0] 
            current_anchor_time_sec = (d_loc / fs)
        else:
            d_loc = None

        # --- Find 'e' wave ---
        e_search_start_time = (current_anchor_time_sec + (0.05 * beat_duration_sec)) if d_loc is not None else ((c_loc / fs if c_loc is not None else (b_loc / fs if b_loc is not None else a_loc / fs)) + (0.15 * beat_duration_sec))
        e_search_end_time = (a_loc / fs) + (0.7 * beat_duration_sec)
        e_candidates = [p for p in all_pos_peaks_global_info if e_search_start_time <= p[0] / fs <= e_search_end_time]
        if e_candidates:
            e_loc = sorted(e_candidates, key=lambda x: (x[0], -x[1]))[0][0] 
        else:
            e_loc = None
    
    return a_loc, b_loc, c_loc, d_loc, e_loc

# --- Main Script Execution ---

if __name__ == "__main__":
    # --- 1. Simulate PPG Signal ---
    sampling_rate = 100 # Hz
    duration = 15     # seconds (Full signal duration for context)
    heart_rate_bpm = 70
    
    time, raw_ppg = simulate_ppg_signal(sampling_rate, duration, heart_rate_bpm)

    # --- 2. Preprocess PPG Signal ---
    filtered_ppg = preprocess_ppg(raw_ppg, sampling_rate)

    # --- 3. Find PPG Systolic Peaks to identify beats ---
    min_peak_distance_ppg = int(sampling_rate * 0.4) 
    prominence_threshold_ppg = 0.1 * np.max(filtered_ppg) 
    
    ppg_systolic_peak_locs, _ = find_peaks(filtered_ppg, distance=min_peak_distance_ppg, prominence=prominence_threshold_ppg)
    
    # Ensure at least two systolic peaks exist to define the first beat
    if len(ppg_systolic_peak_locs) < 2:
        print("Not enough PPG peaks detected to identify the first beat.")
        exit()

    # Define the first beat segment indices (global)
    first_beat_start_idx = ppg_systolic_peak_locs[0]
    first_beat_end_idx = ppg_systolic_peak_locs[1]
    
    # --- 4. Derive APG Signal ---
    vpg_signal = calculate_derivative(filtered_ppg, time[:len(filtered_ppg)])
    apg_signal = calculate_derivative(vpg_signal, time[:len(vpg_signal)])

    # --- 5. Find APG Fiducials for the First Beat ---
    first_beat_a_loc, first_beat_b_loc, first_beat_c_loc, first_beat_d_loc, first_beat_e_loc = \
        find_apg_fiducials_for_single_segment(apg_signal, first_beat_start_idx, first_beat_end_idx, sampling_rate)

    # --- 6. Prepare data for plotting ONLY the first beat APG ---
    # Slice the APG signal and its time array to cover only the first beat
    apg_first_beat_time = time[first_beat_start_idx:first_beat_end_idx+1]
    apg_first_beat_signal = apg_signal[first_beat_start_idx:first_beat_end_idx+1]

    plt.figure(figsize=(14, 10))

    # Plot 1: Full Filtered PPG with First Beat Highlight
    plt.subplot(2, 1, 1)
    plt.plot(time, filtered_ppg, label='Filtered PPG (Full Signal)', color='blue')
    plt.plot(time[ppg_systolic_peak_locs], filtered_ppg[ppg_systolic_peak_locs], 'ro', markersize=5, label='PPG Systolic Peaks')
    
    # Highlight the first beat segment
    plt.axvspan(time[first_beat_start_idx], time[first_beat_end_idx], color='red', alpha=0.1, label='First Beat Segment')
    
    plt.title('Filtered PPG Signal with First Beat Highlighted')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot 2: APG Signal - First Beat Only with Fiducial Points
    plt.subplot(2, 1, 2)
    plt.plot(apg_first_beat_time, apg_first_beat_signal, label='APG Signal (First Beat Only)', color='purple', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7) # Zero line for reference

    # Plot the detected APG fiducial points for the first beat
    # Note: These indices are global, so we use `time[idx]` for correct x-axis plotting.
    if first_beat_a_loc is not None: 
        plt.plot(time[first_beat_a_loc], apg_signal[first_beat_a_loc], 'go', markersize=8, label='\'a\' wave')
    if first_beat_b_loc is not None:
        plt.plot(time[first_beat_b_loc], apg_signal[first_beat_b_loc], 'bo', markersize=8, label='\'b\' wave')
    if first_beat_c_loc is not None:
        plt.plot(time[first_beat_c_loc], apg_signal[first_beat_c_loc], 'co', markersize=8, label='\'c\' wave')
    if first_beat_d_loc is not None:
        plt.plot(time[first_beat_d_loc], apg_signal[first_beat_d_loc], 'mo', markersize=8, label='\'d\' wave')
    if first_beat_e_loc is not None:
        plt.plot(time[first_beat_e_loc], apg_signal[first_beat_e_loc], 'yo', markersize=8, label='\'e\' wave')
            
    plt.title('APG Signal with Fiducial Points (First Beat Only)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() 
    plt.show() 

    print("\n--- First Beat APG Fiducial Points Details ---")
    if first_beat_a_loc is not None:
        print(f"'a' wave: Index = {first_beat_a_loc}, Amplitude = {apg_signal[first_beat_a_loc]:.4f}")
    else:
        print("'a' wave: Not detected for the first beat.")
    
    if first_beat_b_loc is not None:
        print(f"'b' wave: Index = {first_beat_b_loc}, Amplitude = {apg_signal[first_beat_b_loc]:.4f}")
    else:
        print("'b' wave: Not detected for the first beat.")
    
    if first_beat_c_loc is not None:
        print(f"'c' wave: Index = {first_beat_c_loc}, Amplitude = {apg_signal[first_beat_c_loc]:.4f}")
    else:
        print("'c' wave: Not detected for the first beat.")
        
    if first_beat_d_loc is not None:
        print(f"'d' wave: Index = {first_beat_d_loc}, Amplitude = {apg_signal[first_beat_d_loc]:.4f}")
    else:
        print("'d' wave: Not detected for the first beat.")
        
    if first_beat_e_loc is not None:
        print(f"'e' wave: Index = {first_beat_e_loc}, Amplitude = {apg_signal[first_beat_e_loc]:.4f}")
    else:
        print("'e' wave: Not detected for the first beat.")