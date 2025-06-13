import numpy as np 
import signal_filesource

SIGNALSOURCE_FILE = 0
SIGNALSOURCE_SERIAL = 1

class SignalBuffer:
    def __init__(self, type, size, data, update_step = 500):
        self.type = type
        self.data = data
        self.size = size
        self.buffer_time = np.zeros(self.size)
        self.buffer_signal = np.zeros(self.size)
        self.source = 0
        self.start_sample = 0
        self.update_step = update_step
        self.end_sample = self.start_sample + self.size

        if self.type == SIGNALSOURCE_FILE:
            self.source = signal_filesource.FileSignalSource(data["Filename"])
            self.buffer_time, self.buffer_signal = self.source.get_data(self.start_sample, self.end_sample)

    def update_buffer(self):
        self.start_sample = self.start_sample + self.update_step
        self.end_sample = self.start_sample + self.size
        self.buffer_time, self.buffer_signal = self.source.get_data(self.start_sample, self.end_sample)