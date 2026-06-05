import os
import numpy as np
import pickle
import json
import math
from reader import InHospitalMortalityReader, PhenotypingReader, LengthOfStayReader, DecompensationReader
from tqdm import tqdm

with open('resources/channel_info.json') as f:
    series_channel_info = json.load(f)

with open('resources/discretizer_config.json') as f:
    series_config = json.load(f)
    id_to_channel = series_config['id_to_channel']
    is_categorical_channel = series_config['is_categorical_channel']
    normal_values = series_config['normal_values']
    possible_values = series_config['possible_values']

def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data

print("Processing Mortality...")
period_length = 48
path = r'd:\实验室\mamba-cde\mimic3-benchmarks\data\in-hospital-mortality'

data_all = []
mask_all = []
label_all = []
name_all = []
for mode in ['train', 'val', 'test']:
    reader = InHospitalMortalityReader(dataset_dir=os.path.join(path, 'train' if mode != 'test' else 'test'),
            listfile=os.path.join(path, mode + '_listfile.csv'), period_length=period_length)
    N = reader.get_number_of_examples()
    for _ in tqdm(range(N), total=N):
        ret = reader.read_next()
        patient = ret["X"]
        name = ret["name"]
        label_all.append(ret["y"])
        name_all.append(name)
        data_patient = np.zeros(shape=(len(id_to_channel), period_length), dtype=np.float32)
        mask_patient = np.zeros(shape=(len(id_to_channel), period_length), dtype=np.float32)
        last_time = -1
        for row in patient:
            time = int(float(row[0]))
            if time == period_length:
                time -= 1
            if time > period_length:
                raise ValueError('This should not happen')
                break
            for index in range(len(row) - 1):
                value = row[index + 1]
                if value == '':
                    # continue
                    if mask_patient[index, time] == 0 and time - last_time > 0:
                        if last_time >= 0:
                            data_patient[index, last_time + 1:time + 1] = data_patient[index, last_time]
                        else:
                            if is_categorical_channel[id_to_channel[index]]:
                                data_patient[index, last_time + 1:time + 1] = series_channel_info[id_to_channel[index]]['values'][normal_values[id_to_channel[index]]]
                            else:
                                data_patient[index, last_time + 1:time + 1] = float(normal_values[id_to_channel[index]])
                else:
                    mask_patient[index, time] = 1
                    if is_categorical_channel[id_to_channel[index]]:
                        data_patient[index, time] = series_channel_info[id_to_channel[index]]['values'][value]
                    else:
                        data_patient[index, time] = float(value)
            last_time = time
        if last_time < period_length - 1:
            data_patient[:, last_time + 1:period_length] = data_patient[:, last_time, None]
        data_all.append(data_patient.transpose(-1, -2))
        mask_all.append(mask_patient.transpose(-1, -2))

data_all = np.array(data_all)
mask_all = np.array(mask_all)
data_all_concat = np.concatenate(data_all, axis=0)
x_masked = np.ma.masked_array(data_all_concat, np.concatenate(mask_all, axis=0) == 0)
mean = np.mean(x_masked, 0)
std = np.std(x_masked, 0)
std[std == 0] = 1.0 # ADDED TO PREVENT DIV BY ZERO, IF ANY
data_normalized = np.where(mask_all == 1, (data_all - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1), 0)

pickle.dump((data_normalized.tolist(), label_all, mask_all.tolist(), name_all), open('mortality_normalized.pkl', 'wb'))
print("Mortality label rate:", sum(label_all) / len(label_all))

print("Processing Phenotyping...")
period_length = 48
path = r'd:\实验室\mamba-cde\mimic3-benchmarks\data\phenotyping'

data_all = []
mask_all = []
label_all = []
name_all = []
for mode in ['train', 'val', 'test']:
    reader = PhenotypingReader(dataset_dir=os.path.join(path, 'train' if mode != 'test' else 'test'),
            listfile=os.path.join(path, mode + '_listfile.csv'))
    N = reader.get_number_of_examples()
    for _ in tqdm(range(N), total=N):
        ret = reader.read_next()
        patient = ret["X"]
        t = ret["t"]
        name = ret["name"]
        label_all.append(ret["y"])
        name_all.append(name)
        N_bins = min(int(t + 1 - 1e-6), period_length)
        data_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        mask_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        last_time = -1
        for row in patient:
            time = int(float(row[0]))
            if time == N_bins:
                time -= 1
            if time > N_bins:
                break
            for index in range(len(row) - 1):
                value = row[index + 1]
                if value == '':
                    if mask_patient[index, time] == 0 and time - last_time > 0:
                        if is_categorical_channel[id_to_channel[index]]:
                            data_patient[index, last_time + 1:time + 1] = series_channel_info[id_to_channel[index]]['values'][normal_values[id_to_channel[index]]]
                        else:
                            data_patient[index, last_time + 1:time + 1] = float(normal_values[id_to_channel[index]])
                else:
                    mask_patient[index, time] += 1
                    if is_categorical_channel[id_to_channel[index]]:
                        data_patient[index, time] += series_channel_info[id_to_channel[index]]['values'][value]
                    else:
                        data_patient[index, time] += float(value)
            last_time = time
        data_patient = np.where(mask_patient > 0, data_patient / mask_patient, data_patient)
        mask_patient = np.where(mask_patient > 0, 1, 0)
        data_all.append(data_patient.transpose(-1, -2))
        mask_all.append(mask_patient.transpose(-1, -2))

data_all_concat = np.concatenate(data_all, axis=0)
mean = np.mean(data_all_concat, 0)
std = np.std(data_all_concat, 0)
std[std == 0] = 1.0
data_normalized = [((d - mean.reshape(1, -1)) / std.reshape(1, -1)).tolist() for d in data_all]
mask_all_list = [m.tolist() for m in mask_all]

pickle.dump((data_normalized, label_all, mask_all_list, name_all), open('phenotyping_normalized.pkl', 'wb'))
print("Phenotyping label rate:", np.sum(label_all) / len(label_all))

print("Processing Decompensation...")
period_length = 24
path = r'd:\实验室\mamba-cde\mimic3-benchmarks\data\decompensation'

data_all = []
mask_all = []
label_all = []
name_all = []
for mode in ['train', 'val', 'test']:
    reader = DecompensationReader(dataset_dir=os.path.join(path, 'train' if mode != 'test' else 'test'),
            listfile=os.path.join(path, mode + '_listfile.csv'))
    N = reader.get_number_of_examples()
    for _ in tqdm(range(N), total=N):
        ret = reader.read_next()
        patient = ret["X"]
        t = ret["t"]
        name = ret["name"]
        label_all.append(ret["y"])
        name_all.append(name)
        N_bins = min(int(t + 1 - 1e-6), period_length)
        data_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        mask_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        last_time = -1
        for row in patient:
            time = int(float(row[0]))
            if time == N_bins:
                time -= 1
            if time > N_bins:
                break
            for index in range(len(row) - 1):
                value = row[index + 1]
                if value == '':
                    if mask_patient[index, time] == 0 and time - last_time > 0:
                        if is_categorical_channel[id_to_channel[index]]:
                            data_patient[index, last_time + 1:time + 1] = series_channel_info[id_to_channel[index]]['values'][normal_values[id_to_channel[index]]]
                        else:
                            data_patient[index, last_time + 1:time + 1] = float(normal_values[id_to_channel[index]])
                else:
                    mask_patient[index, time] += 1
                    if is_categorical_channel[id_to_channel[index]]:
                        data_patient[index, time] += series_channel_info[id_to_channel[index]]['values'][value]
                    else:
                        data_patient[index, time] += float(value)
            last_time = time
        with np.errstate(divide='ignore', invalid='ignore'):
            data_patient = np.where(mask_patient > 0, data_patient / mask_patient, data_patient)
        mask_patient = np.where(mask_patient > 0, 1, 0)
        data_all.append(data_patient.transpose(-1, -2))
        mask_all.append(mask_patient.transpose(-1, -2))

data_all_concat = np.concatenate(data_all, axis=0)
mean = np.mean(data_all_concat, 0)
std = np.std(data_all_concat, 0)
std[std == 0] = 1.0
data_normalized = [((d - mean.reshape(1, -1)) / std.reshape(1, -1)).tolist() for d in data_all]
mask_all_list = [m.tolist() for m in mask_all]

pickle.dump((data_normalized, label_all, mask_all_list, name_all), open('decompensation_normalized.pkl', 'wb'))
print("Decompensation label rate:", sum(label_all) / len(label_all))

print("Processing Length of Stay...")
period_length = 24
path = r'd:\实验室\mamba-cde\mimic3-benchmarks\data\length-of-stay'

data_all = []
mask_all = []
label_all = []
name_all = []
for mode in ['train', 'val', 'test']:
    reader = LengthOfStayReader(dataset_dir=os.path.join(path, 'train' if mode != 'test' else 'test'),
            listfile=os.path.join(path, mode + '_listfile.csv'))
    N = reader.get_number_of_examples()
    for _ in tqdm(range(N), total=N):
        ret = reader.read_next()
        patient = ret["X"]
        t = ret["t"]
        name = ret["name"]
        label_all.append(ret["y"])
        name_all.append(name)
        N_bins = min(int(t + 1 - 1e-6), period_length)
        data_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        mask_patient = np.zeros(shape=(len(id_to_channel), N_bins), dtype=np.float32)
        last_time = -1
        for row in patient:
            time = int(float(row[0]))
            if time == N_bins:
                time -= 1
            if time > N_bins:
                break
            for index in range(len(row) - 1):
                value = row[index + 1]
                if value == '':
                    if mask_patient[index, time] == 0 and time - last_time > 0:
                        if is_categorical_channel[id_to_channel[index]]:
                            data_patient[index, last_time + 1:time + 1] = series_channel_info[id_to_channel[index]]['values'][normal_values[id_to_channel[index]]]
                        else:
                            data_patient[index, last_time + 1:time + 1] = float(normal_values[id_to_channel[index]])
                else:
                    mask_patient[index, time] += 1
                    if is_categorical_channel[id_to_channel[index]]:
                        data_patient[index, time] += series_channel_info[id_to_channel[index]]['values'][value]
                    else:
                        data_patient[index, time] += float(value)
            last_time = time
        with np.errstate(divide='ignore', invalid='ignore'):
            data_patient = np.where(mask_patient > 0, data_patient / mask_patient, data_patient)
        mask_patient = np.where(mask_patient > 0, 1, 0)
        data_all.append(data_patient.transpose(-1, -2))
        mask_all.append(mask_patient.transpose(-1, -2))

data_all_concat = np.concatenate(data_all, axis=0)
mean = np.mean(data_all_concat, 0)
std = np.std(data_all_concat, 0)
std[std == 0] = 1.0
data_normalized = [((d - mean.reshape(1, -1)) / std.reshape(1, -1)).tolist() for d in data_all]
mask_all_list = [m.tolist() for m in mask_all]

# NOTE: The original script doesn't convert labels/24 for classification?
# Actually in the original, lengthofstay task doesn't have a label conversion loop in the ipynb.
# Let's check: "label_all = np.array(label_all)" 

pickle.dump((data_normalized, label_all, mask_all_list, name_all), open('lengthofstay_normalized.pkl', 'wb'))
print("Length of stay completion!")
