# Model for a signal source - but the source is a static file (dataset)

import pandas as pd

class PPGSignal:
    def __init__(self, time_values, signal_values, sampling_period, metadata=None):
        self.time_values = time_values
        self.signal_values = signal_values
        self.sampling_period = sampling_period
        self.sampling_frequency = 1 / sampling_period
        self.metadata = metadata
    
    def __repr__(self):
        return (f"PPGSignal(Sampling Frequency: {self.sampling_frequency} Hz, "
                f"Samples: {len(self.time_values)}, Metadata: {self.metadata})")
    
    def get_data(self, start_sample, end_sample):
        if start_sample < 0:
            start_sample = 0
        if end_sample >= len(self.signal_values):
            end_sample = len(self.signal_values)
            
        return self.time_values[start_sample: end_sample], self.signal_values[start_sample: end_sample]

class FileSignalSource:
    def __init__(self, filename):
        self.filename = filename
        self.signal = 0

        self.load_signal()

    def load_signal(self):
        data = pd.read_csv(self.filename)
        time_values = data['Time'].values  
        signal_values = data['PPG'].values  
        sampling_period = time_values[1] - time_values[0]  
        self.signal = PPGSignal(time_values, signal_values, sampling_period, metadata="File Source")

    def get_data(self, start_sample, end_sample):
        return self.signal.get_data(start_sample, end_sample)
    
if __name__ == "__main__":
    testfs = FileSignalSource("Datasets/Set-1/data_ppg.csv")
    print(testfs)