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

        apg = calculate_derivative(calculate_derivative(s,t),t)

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

