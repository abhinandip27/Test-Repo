# Processing of the Plethsymograph Signal

from scipy.signal import find_peaks
from preprocessing import preprocess_ppg,calculate_derivative
import numpy as np
import queue

class PlethSignalProcessor:
    def __init__(self, signalbuffer, featurebuffer_maxsize = 50):
        self.signalbuffer = signalbuffer
        self.s_loc,self.di_loc,self.o_loc,self.n_loc = 0,0,0,0

        self.hr = 0
        self.avgs = 0
        self.AI = 0
        self.hrv = 0
        self.hr_queue = queue.Queue(featurebuffer_maxsize)
        self.avgs_queue = queue.Queue(featurebuffer_maxsize)
        self.AI_queue = queue.Queue(featurebuffer_maxsize)
        self.hrv_queue = queue.Queue(featurebuffer_maxsize)

    def get_feature(self):
        self.find_fiducial()
        self.find_hr()
        self.find_amplitudefeatures()
        self.find_Augmentationindex()
        self.find_hrv()

        if self.hr_queue.full():
            self.hr_queue.get()
        self.hr_queue.put(self.hr)

        if self.avgs_queue.full():
            self.avgs_queue.get()
        self.avgs_queue.put(self.avgs)

        if self.AI_queue.full():
            self.AI_queue.get()
        self.AI_queue.put(self.AI)

        if self.hrv_queue.full():
            self.hrv_queue.get()
        self.hrv_queue.put(self.hrv)

        return {"HR": self.hr, "SysA": self.avgs, "AI": self.AI, "HRV": self.hrv}
    
    def get_feature_series(self):
        return {"HR": self.hr_queue, "SysA": self.avgs_queue, "AI": self.AI_queue, "HRV": self.hrv_queue}
    
    def find_fiducial(self):
        t = self.signalbuffer.buffer_time

        s = preprocess_ppg(self.signalbuffer.buffer_signal)
        peakloc_p = find_peaks(s)[0]
        peakloc_n = find_peaks(-s)[0]
        p_mu = np.mean(s[peakloc_p])
        n_mu = np.mean(s[peakloc_n])
        self.s_loc = peakloc_p[np.where(s[peakloc_p] > p_mu)[0]]
        self.di_loc = peakloc_p[np.where(s[peakloc_p] <= p_mu)[0]]
        self.o_loc = peakloc_n[np.where(s[peakloc_n] < n_mu)[0]]
        self.n_loc = peakloc_n[np.where(s[peakloc_n] >= n_mu)[0]]

        apg=calculate_derivative(calculate_derivative(s,t),t)
        a,b,c,d,e=find_apg_fiducials(apg,self.s_loc[0],self.s_loc[1],self.signalbuffer.size)
        aging_index = (b - c - d - e) / a if a != 0 else 0
        apg_features = {
            "AI": aging_index,
            "b/a": b/a, "c/a": c/a, "d/a": d/a, "e/a": e/a
        }


    def find_hr(self):
        t = self.signalbuffer.buffer_time
        ts = t[self.s_loc]
        self.hr = [ts[-1], 60 * 1/np.mean(np.diff(ts))]
    
    def find_hrv(self):
        if len(self.s_loc) < 2:
          self.hrv = 0
          return

        t = self.signalbuffer.buffer_time
        ts = t[self.s_loc]

        rr_intervals = np.diff(ts)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) 
        
        self.hrv = [ts[-1],rmssd]

    def find_Augmentationindex(self):
        s_values = self.signalbuffer.buffer_signal[self.s_loc]
        di_values = self.signalbuffer.buffer_signal[self.di_loc]
        t = self.signalbuffer.buffer_time
        ts = t[self.s_loc]

        min_len = min(len(s_values), len(di_values))
        if min_len > 0:
            s_aligned = s_values[:min_len]
            di_aligned = di_values[:min_len]
            self.AI = [ts[-1],np.mean((1 - di_aligned / s_aligned) * 100)]
        else:
            self.AI = 0

    def find_amplitudefeatures(self):
        t = self.signalbuffer.buffer_time[self.s_loc][-1]
        s = self.signalbuffer.buffer_signal[self.s_loc]
        self.avgs = [t, np.mean(s)]
        

def find_apg_fiducials(apg_signal, beat_start_idx, beat_end_idx, fs):

    apg_beat_segment = apg_signal[beat_start_idx:beat_end_idx]

    a_loc, b_loc, c_loc, d_loc, e_loc = None, None, None, None, None

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
        c_search_start_time = current_anchor_time_sec + (0.05 * beat_duration_sec) 
        c_search_end_time = (a_loc / fs) + (0.5 * beat_duration_sec)
        c_candidates = [p for p in all_pos_peaks_global_info if c_search_start_time <= p[0] / fs <= c_search_end_time]
        if c_candidates:
            c_loc = sorted(c_candidates, key=lambda x: (x[0], -x[1]))[0][0] 
            current_anchor_time_sec = (c_loc / fs)
        else:
            c_loc = None 

        d_search_start_time = (current_anchor_time_sec + (0.05 * beat_duration_sec)) if c_loc is not None else ((b_loc / fs if b_loc is not None else a_loc / fs) + (0.1 * beat_duration_sec))
        d_search_end_time = (a_loc / fs) + (0.6 * beat_duration_sec)
        d_candidates = [p for p in all_neg_peaks_global_info if d_search_start_time <= p[0] / fs <= d_search_end_time]
        if d_candidates:
            d_loc = sorted(d_candidates, key=lambda x: (x[0], x[1]))[0][0] 
            current_anchor_time_sec = (d_loc / fs)
        else:
            d_loc = None

        e_search_start_time = (current_anchor_time_sec + (0.05 * beat_duration_sec)) if d_loc is not None else ((c_loc / fs if c_loc is not None else (b_loc / fs if b_loc is not None else a_loc / fs)) + (0.15 * beat_duration_sec))
        e_search_end_time = (a_loc / fs) + (0.7 * beat_duration_sec)
        e_candidates = [p for p in all_pos_peaks_global_info if e_search_start_time <= p[0] / fs <= e_search_end_time]
        if e_candidates:
            e_loc = sorted(e_candidates, key=lambda x: (x[0], -x[1]))[0][0] 
        else:
            e_loc = None

    return apg_signal[a_loc],apg_signal[ b_loc], apg_signal[c_loc], apg_signal[d_loc], apg_signal[e_loc]