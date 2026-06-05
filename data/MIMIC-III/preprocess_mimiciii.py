"""
Preprocess MIMIC-III data for 4 tasks:
  - In-Hospital Mortality  -> mortality_normalized.pkl
  - Phenotyping            -> phenotyping_normalized.pkl
  - Decompensation         -> decompensation_normalized.pkl
  - Length of Stay         -> lengthofstay_normalized.pkl

Prerequisites (run ONCE before this script):
  1. Clone mimic3-benchmarks and follow its setup guide to produce the
     per-patient CSVs from the raw MIMIC-III PostgreSQL dump.
       https://github.com/YerevaNN/mimic3-benchmarks
  2. Run the benchmark extraction scripts:
       python -m mimic3benchmark.scripts.extract_subjects <MIMIC_CSV_DIR> data/root
       python -m mimic3benchmark.scripts.validate_events data/root
       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root
       python -m mimic3benchmark.scripts.split_train_and_test data/root
  3. Run the task-creation scripts (from mimic3-benchmarks root):
       python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root data/in-hospital-mortality
       python -m mimic3benchmark.scripts.create_phenotyping          data/root data/phenotyping
  4. For decompensation and length-of-stay, copy SMART's helper scripts into
     the mimic3-benchmarks folder and run:
       python create_decompensation.py  data/root data/decompensation
       python create_length_of_stay.py  data/root data/length-of-stay
  5. Place (or symlink) those output folders here:
       SMART/data/MIMIC-III/in-hospital-mortality/
       SMART/data/MIMIC-III/phenotyping/
       SMART/data/MIMIC-III/decompensation/
       SMART/data/MIMIC-III/length-of-stay/
     Also copy the resources/ folder from mimic3-benchmarks:
       SMART/data/MIMIC-III/resources/

Run (from SMART/):
    cd data/MIMIC-III
    python preprocess_mimiciii.py
    # or selectively:
    python preprocess_mimiciii.py --tasks mortality phenotyping
"""
import os
import sys
import json
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from reader import (
    InHospitalMortalityReader,
    PhenotypingReader,
    DecompensationReader,
    LengthOfStayReader,
)

# ---------------------------------------------------------------------------
# read_chunk: batch-read N examples from a Reader
# ---------------------------------------------------------------------------
def read_chunk(reader, chunk_size):
    data = {}
    for _ in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


# ---------------------------------------------------------------------------
# Load discretizer config (from resources/)
# ---------------------------------------------------------------------------
def load_channel_config(resources_dir='resources'):
    with open(os.path.join(resources_dir, 'channel_info.json')) as f:
        channel_info = json.load(f)
    with open(os.path.join(resources_dir, 'discretizer_config.json')) as f:
        config = json.load(f)
    id_to_channel          = config['id_to_channel']
    is_categorical_channel = config['is_categorical_channel']
    normal_values          = config['normal_values']
    possible_values        = config['possible_values']
    return channel_info, id_to_channel, is_categorical_channel, normal_values


# ---------------------------------------------------------------------------
# Core processor: convert raw Reader rows -> (T, C) arrays + mask
# ---------------------------------------------------------------------------
def process_patient(patient_rows, n_channels, period_length,
                    channel_info, id_to_channel,
                    is_categorical_channel, normal_values):
    data_patient = np.zeros((n_channels, period_length), dtype=np.float32)
    mask_patient = np.zeros((n_channels, period_length), dtype=np.float32)
    last_time = -1
    for row in patient_rows:
        time = int(float(row[0]))
        if time == period_length:
            time -= 1
        if time >= period_length:
            break
        for idx in range(len(row) - 1):
            value = row[idx + 1]
            ch = id_to_channel[idx]
            if value == '':
                if mask_patient[idx, time] == 0 and time - last_time > 0:
                    if is_categorical_channel[ch]:
                        fill = channel_info[ch]['values'][normal_values[ch]]
                    else:
                        fill = float(normal_values[ch])
                    start = max(0, last_time + 1)
                    data_patient[idx, start:time + 1] = fill
            else:
                mask_patient[idx, time] += 1
                if is_categorical_channel[ch]:
                    data_patient[idx, time] += channel_info[ch]['values'][value]
                else:
                    data_patient[idx, time] += float(value)
        last_time = time

    # average multiple observations in same time bin
    with np.errstate(divide='ignore', invalid='ignore'):
        data_patient = np.where(
            mask_patient > 0, data_patient / mask_patient, data_patient
        )
    mask_patient = np.where(mask_patient > 0, 1, 0)
    return data_patient.T, mask_patient.T   # (T, C)


def observed_rate(mask_list):
    cnt = cnt1 = 0
    for m in mask_list:
        for row in m:
            cnt  += sum(row)
            cnt1 += len(row)
    return cnt / cnt1 if cnt1 > 0 else 0.0


# ---------------------------------------------------------------------------
# Task 1: In-Hospital Mortality
# ---------------------------------------------------------------------------
def process_mortality(base_path, channel_info, id_to_channel,
                      is_categorical_channel, normal_values, out_dir):
    period_length = 48
    n_channels    = len(id_to_channel)
    path          = os.path.join(base_path, 'in-hospital-mortality')

    data_all  = []
    mask_all  = []
    label_all = []
    name_all  = []
    split_sizes = []

    for mode in ['train', 'val', 'test']:
        split_dir = 'train' if mode != 'test' else 'test'
        reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(path, split_dir),
            listfile=os.path.join(path, mode + '_listfile.csv'),
            period_length=period_length,
        )
        N   = reader.get_number_of_examples()
        ret = read_chunk(reader, N)
        label_all += ret['y']
        name_all  += ret['name']
        for patient_rows in tqdm(ret['X'], desc=f'Mortality {mode}'):
            dp, mp = process_patient(
                patient_rows, n_channels, period_length,
                channel_info, id_to_channel, is_categorical_channel, normal_values
            )
            # forward-fill to end
            last_t = dp.shape[0] - 1
            data_all.append(dp)
            mask_all.append(mp)
        split_sizes.append(N)

    data_arr = np.array(data_all)             # (N, T, C)
    mask_arr = np.array(mask_all)

    # normalize on all data (matches paper/notebook)
    flat_data = data_arr.reshape(-1, n_channels)
    flat_mask = mask_arr.reshape(-1, n_channels)
    masked    = np.ma.masked_array(flat_data, flat_mask == 0)
    mean      = np.mean(masked, 0)
    std       = np.std(masked, 0)
    std[std == 0] = 1.0
    data_norm = np.where(mask_arr == 1,
                         (data_arr - mean) / std, 0.0)

    print(f'Mortality: {len(label_all)} samples {split_sizes}, '
          f'pos_rate={sum(label_all)/len(label_all):.4f}, '
          f'obs_rate={observed_rate([m.tolist() for m in mask_all]):.4f}')

    pkl_path = os.path.join(out_dir, 'mortality_normalized.pkl')
    pickle.dump(
        (data_norm.tolist(), label_all, mask_arr.tolist(), name_all, split_sizes),
        open(pkl_path, 'wb')
    )
    print(f'Saved: {pkl_path}')


# ---------------------------------------------------------------------------
# Task 2: Phenotyping
# ---------------------------------------------------------------------------
def process_phenotyping(base_path, channel_info, id_to_channel,
                        is_categorical_channel, normal_values, out_dir):
    period_length = 48
    n_channels    = len(id_to_channel)
    path          = os.path.join(base_path, 'phenotyping')

    data_all  = []
    mask_all  = []
    label_all = []
    name_all  = []
    split_sizes = []

    for mode in ['train', 'val', 'test']:
        split_dir = 'train' if mode != 'test' else 'test'
        reader = PhenotypingReader(
            dataset_dir=os.path.join(path, split_dir),
            listfile=os.path.join(path, mode + '_listfile.csv'),
        )
        N   = reader.get_number_of_examples()
        ret = read_chunk(reader, N)
        label_all += ret['y']
        name_all  += ret['name']
        for patient_rows, t in tqdm(zip(ret['X'], ret['t']), desc=f'Pheno {mode}', total=N):
            n_bins = min(int(t + 1 - 1e-6), period_length)
            dp, mp = process_patient(
                patient_rows, n_channels, n_bins,
                channel_info, id_to_channel, is_categorical_channel, normal_values
            )
            data_all.append(dp)
            mask_all.append(mp)
        split_sizes.append(N)

    # normalize on all data (matches paper/notebook)
    all_concat = np.concatenate(data_all, axis=0)
    mean       = np.mean(all_concat, 0)
    std        = np.std(all_concat, 0)
    std[std == 0] = 1.0

    data_norm = [((d - mean) / std).tolist() for d in data_all]
    mask_list = [m.tolist() for m in mask_all]

    label_arr = np.array(label_all)
    print(f'Phenotyping: {len(label_all)} samples {split_sizes}, '
          f'pos_rate={np.mean(label_arr, 0).mean():.4f}, '
          f'obs_rate={observed_rate(mask_list):.4f}')

    pkl_path = os.path.join(out_dir, 'phenotyping_normalized.pkl')
    pickle.dump((data_norm, label_all, mask_list, name_all, split_sizes), open(pkl_path, 'wb'))
    print(f'Saved: {pkl_path}')


# ---------------------------------------------------------------------------
# Task 3: Decompensation  (streaming to avoid OOM on raw timeseries)
# ---------------------------------------------------------------------------
def process_decompensation(base_path, channel_info, id_to_channel,
                           is_categorical_channel, normal_values, out_dir):
    period_length = 24
    n_channels    = len(id_to_channel)
    # Use SMART's one-per-stay format (period_length<=24, generated by
    # create_decompensation.py) rather than the standard benchmark's sliding-
    # window format where period_length can exceed 2000h.  With the standard
    # format, preprocess_mimiciii.py truncates all sequences to 24 bins but
    # the label refers to decompensation at hour t (median ~64h), creating a
    # severe feature/label mismatch.
    _mimic3_base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '..', 'mimic3-benchmarks', 'data'))
    smart_path = os.path.join(_mimic3_base, 'decompensation_smart')
    std_path   = os.path.join(base_path, 'decompensation')
    if os.path.isdir(smart_path) and os.path.exists(
            os.path.join(smart_path, 'train_listfile.csv')):
        path = smart_path
        print('Decompensation: using SMART one-per-stay format (decompensation_smart/)')
    else:
        path = std_path
        print('WARNING: decompensation_smart/ not found, falling back to standard '
              'benchmark format. Run create_decompensation.py + make_listfiles.py '
              'to generate the correct data (see make_listfiles.py for instructions).')

    data_all  = []
    mask_all  = []
    label_all = []
    name_all  = []
    split_sizes = []

    for mode in ['train', 'val', 'test']:
        split_dir = 'train' if mode != 'test' else 'test'
        reader = DecompensationReader(
            dataset_dir=os.path.join(path, split_dir),
            listfile=os.path.join(path, mode + '_listfile.csv'),
        )
        N = reader.get_number_of_examples()
        for _ in tqdm(range(N), desc=f'Decomp {mode}'):
            ret = reader.read_next()
            t   = ret['t']
            n_bins = min(int(t + 1 - 1e-6), period_length)
            dp, mp = process_patient(
                ret['X'], n_channels, n_bins,
                channel_info, id_to_channel, is_categorical_channel, normal_values
            )
            data_all.append(dp)
            mask_all.append(mp)
            label_all.append(ret['y'])
            name_all.append(ret['name'])
        split_sizes.append(N)

    # normalize on all data (matches paper/notebook)
    all_concat = np.concatenate(data_all, axis=0)
    mean       = np.mean(all_concat, 0)
    std        = np.std(all_concat, 0)
    std[std == 0] = 1.0
    del all_concat

    data_norm = [(d - mean) / std for d in data_all]

    print(f'Decompensation: {len(label_all)} samples {split_sizes}, '
          f'pos_rate={sum(label_all)/len(label_all):.4f}, '
          f'obs_rate={observed_rate([m.tolist() for m in mask_all]):.4f}')

    pkl_path = os.path.join(out_dir, 'decompensation_normalized.pkl')
    pickle.dump((data_norm, label_all, mask_all, name_all, split_sizes), open(pkl_path, 'wb'))
    print(f'Saved: {pkl_path}')


# ---------------------------------------------------------------------------
# Task 4: Length of Stay  (streaming to avoid OOM on raw timeseries)
# ---------------------------------------------------------------------------
def process_lengthofstay(base_path, channel_info, id_to_channel,
                         is_categorical_channel, normal_values, out_dir):
    period_length = 24
    n_channels    = len(id_to_channel)
    path          = os.path.join(base_path, 'length-of-stay')

    data_all  = []
    mask_all  = []
    label_all = []
    name_all  = []
    split_sizes = []

    for mode in ['train', 'val', 'test']:
        split_dir = 'train' if mode != 'test' else 'test'
        reader = LengthOfStayReader(
            dataset_dir=os.path.join(path, split_dir),
            listfile=os.path.join(path, mode + '_listfile.csv'),
        )
        N = reader.get_number_of_examples()
        for _ in tqdm(range(N), desc=f'LoS {mode}'):
            ret = reader.read_next()
            t   = ret['t']
            n_bins = min(int(t + 1 - 1e-6), period_length)
            dp, mp = process_patient(
                ret['X'], n_channels, n_bins,
                channel_info, id_to_channel, is_categorical_channel, normal_values
            )
            data_all.append(dp)
            mask_all.append(mp)
            label_all.append(ret['y'])
            name_all.append(ret['name'])
        split_sizes.append(N)

    # normalize on all data (matches paper/notebook)
    all_concat = np.concatenate(data_all, axis=0)
    mean       = np.mean(all_concat, 0)
    std        = np.std(all_concat, 0)
    std[std == 0] = 1.0
    del all_concat

    # convert hours -> days; keep as numpy arrays to save memory
    label_all_days = [[y / 24] for y in label_all]
    data_norm = [(d - mean) / std for d in data_all]

    print(f'LengthOfStay: {len(label_all)} samples {split_sizes}, '
          f'obs_rate={observed_rate([m.tolist() for m in mask_all]):.4f}')

    pkl_path = os.path.join(out_dir, 'lengthofstay_normalized.pkl')
    pickle.dump((data_norm, label_all_days, mask_all, name_all, split_sizes), open(pkl_path, 'wb'))
    print(f'Saved: {pkl_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_path', type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help='Root directory containing in-hospital-mortality/, phenotyping/, etc.'
    )
    parser.add_argument(
        '--resources', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources'),
        help='Path to resources/ directory with channel_info.json etc.'
    )
    parser.add_argument(
        '--out_dir', type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help='Output directory for .pkl files'
    )
    parser.add_argument(
        '--tasks', nargs='+',
        default=['mortality', 'phenotyping', 'decompensation', 'lengthofstay'],
        choices=['mortality', 'phenotyping', 'decompensation', 'lengthofstay'],
        help='Which tasks to process'
    )
    args = parser.parse_args()

    # Check resources
    for fname in ['channel_info.json', 'discretizer_config.json']:
        fpath = os.path.join(args.resources, fname)
        if not os.path.exists(fpath):
            sys.exit(f'ERROR: {fpath} not found. '
                     f'Copy resources/ from mimic3-benchmarks repo.')

    channel_info, id_to_channel, is_categorical_channel, normal_values = \
        load_channel_config(args.resources)

    print(f'Channels: {len(id_to_channel)}')

    task_map = {
        'mortality':       process_mortality,
        'phenotyping':     process_phenotyping,
        'decompensation':  process_decompensation,
        'lengthofstay':    process_lengthofstay,
    }

    for task in args.tasks:
        print(f'\n{"="*60}')
        print(f'Processing: {task}')
        print(f'{"="*60}')
        task_map[task](
            args.base_path, channel_info, id_to_channel,
            is_categorical_channel, normal_values, args.out_dir
        )

    print('\nAll done.')


if __name__ == '__main__':
    main()
