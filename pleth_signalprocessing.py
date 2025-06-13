# Processing of the Plethsymograph Signal

from scipy.signal import find_peaks
from preprocessing import preprocess_ppg,calculate_derivative
import numpy as np
import queue

class PlethSignalProcessor:
    def __init__(self, signalbuffer, featurebuffer_maxsize = 50):
        self.signalbuffer = signalbuffer
        self.s_loc,self.di_loc,self.o_loc,self.n_loc = 0,0,0,0
        self.w_loc,self.x_loc,self.y_loc,self.z_loc = 0,0,0,0
        self.apg_a_locs = []
        self.apg_b_locs = []
        self.apg_c_locs = []
        self.apg_d_locs = []
        self.apg_e_locs = []

        self.hr = 0
        self.avgs = 0
        self.a,self.b,self.c,self.d,self.e=0,0,0,0,0
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

        self.apg = calculate_derivative(calculate_derivative(s,t),t)

        self._find_apg_fiducial_points()


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

    def _find_apg_fiducial_points(self):
        self.apg_a_locs = []
        self.apg_b_locs = []
        self.apg_c_locs = []
        self.apg_d_locs = []
        self.apg_e_locs = []

        if len(self.s_loc) < 2:
            return # Need at least two PPG systolic peaks to define a beat segment

        for i in range(len(self.s_loc) - 1):
            beat_start_idx = self.s_loc[i]
            beat_end_idx = self.s_loc[i+1]

            if beat_start_idx >= len(self.apg) or beat_end_idx > len(self.apg) or beat_start_idx >= beat_end_idx:
                continue

            apg_beat_segment = self.apg[beat_start_idx:beat_end_idx]
            
            a_loc, b_loc, c_loc, d_loc, e_loc = None, None, None, None, None

            pos_peaks_local_idx = find_peaks(apg_beat_segment)[0]
            neg_peaks_local_idx = find_peaks(-apg_beat_segment)[0]

            all_critical_points_local_idx = np.sort(
                np.concatenate((pos_peaks_local_idx, neg_peaks_local_idx))
            )

            current_state = 'expect_a'
            for local_idx in all_critical_points_local_idx:
                global_idx = beat_start_idx + local_idx
                apg_value = apg_beat_segment[local_idx]

                if current_state == 'expect_a' and apg_value > 0 and local_idx in pos_peaks_local_idx:
                    a_loc = global_idx
                    current_state = 'expect_b'
                elif current_state == 'expect_b' and apg_value < 0 and local_idx in neg_peaks_local_idx:
                    b_loc = global_idx
                    current_state = 'expect_c'
                elif current_state == 'expect_c' and apg_value > 0 and local_idx in pos_peaks_local_idx:
                    c_loc = global_idx
                    current_state = 'expect_d'
                elif current_state == 'expect_d' and apg_value < 0 and local_idx in neg_peaks_local_idx:
                    d_loc = global_idx
                    current_state = 'expect_e'
                elif current_state == 'expect_e' and apg_value > 0 and local_idx in pos_peaks_local_idx:
                    e_loc = global_idx
                    break # Found all primary points for this beat

            if a_loc is not None:
                self.apg_a_locs.append(a_loc)
                self.apg_b_locs.append(b_loc)
                self.apg_c_locs.append(c_loc)
                self.apg_d_locs.append(d_loc)
                self.apg_e_locs.append(e_loc)

        # For single-beat features like AI, you might want to use the latest detected beat's APG points.
        # This will take the points from the most recently processed beat.
        if self.apg_a_locs:
            self.a = self.apg[self.apg_a_locs[-1]] if self.apg_a_locs[-1] is not None else 0
            self.b = self.apg[self.apg_b_locs[-1]] if self.apg_b_locs[-1] is not None else 0
            self.c = self.apg[self.apg_c_locs[-1]] if self.apg_c_locs[-1] is not None else 0
            self.d = self.apg[self.apg_d_locs[-1]] if self.apg_d_locs[-1] is not None else 0
            self.e = self.apg[self.apg_e_locs[-1]] if self.apg_e_locs[-1] is not None else 0
        else:
            self.a, self.b, self.c, self.d, self.e = 0, 0, 0, 0, 0

    
    def apg_feature(self):
        aging_index = (self.b - self.c - self.d - self.e) / self.a if self.a != 0 else 0
        self.apg_features = {
            "AI": aging_index,
            "b/a": self.b/self.a, "c/a": self.c/self.a, "d/a": self.d/self.a, "e/a": self.e/self.a
        }

