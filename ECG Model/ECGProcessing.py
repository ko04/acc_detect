# Relevant links: 
# https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/quickstart.html#basic-example
# https://linuxpip.org/fix-python-unresolved-import-in-vscode/
import heartpy as hp

print("Testing...")
hrdata = hp.get_data('2022-02-23_LLC_0001.csv')
data, _ = hp.load_exampledata(0)
working_data, measures = hp.process(data, 100.0)
print(measures['bpm']) #returns BPM value
print(measures['rmssd']) # returns RMSSD HRV measure
print("All good!")