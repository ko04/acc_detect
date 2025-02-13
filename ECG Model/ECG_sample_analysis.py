#The NeuroKit API, should allow us to not only pre-process and clean the data, but also return important values to us
#such as R-Peaks, sample entropy, etc.
#Link to NeuroKit API GitHub: https://github.com/neuropsychology/NeuroKit

import numpy as np
import pandas as pd
import neurokit2 as nk

print("Starting...")
# Generate synthetic signals
ecg = nk.ecg_simulate(duration=10, heart_rate=70)
ppg = nk.ppg_simulate(duration=10, heart_rate=70)
rsp = nk.rsp_simulate(duration=10, respiratory_rate=15)
eda = nk.eda_simulate(duration=10, scr_number=3)
emg = nk.emg_simulate(duration=10, burst_number=2)

# Visualise biosignals
data = pd.DataFrame({"ECG": ecg,
                     "PPG": ppg,
                     "RSP": rsp,
                     "EDA": eda,
                     "EMG": emg})
nk.signal_plot(data, subplots=True)
print("All good!")