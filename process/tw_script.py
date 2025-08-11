#!/usr/bin/env python3
import os
import shutil
import numpy as np
from obspy import read
import argparse

search_range = 10

data_directory = "/Users/ahasan/data/RF/na/wabash/grid.jinv2024/"
target_directory = data_directory
file_path = "/Users/ahasan/paper/others/LiuYC/pmp/tpmp.xy"
base_directory = "/Users/ahasan/data/SSPMP/wabash/"


def extract_numerical_value(folder_name):
    try:
        folder_name = folder_name.replace("grid", "")  
        return int(folder_name)
    except ValueError:
        return None

def read_columns(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        first_column = [float(line.split()[0]) for line in lines if len(line.split()) > 0]
        third_column = [line.split()[2] for line in lines if len(line.split()) > 2]
        fourth_column = [line.split()[3] for line in lines if len(line.split()) > 3]
    return first_column, third_column, fourth_column

def calculate_rms_noise(data, sampling_rate):
    max_index = np.argmax(np.abs(data))
    end_time_before_max = max_index - int(20 * sampling_rate)
    if end_time_before_max > 0:
        data_before_max = data[:end_time_before_max]
        if not np.isnan(data_before_max).any() and not np.isinf(data_before_max).any():
            return np.sqrt(np.mean(data_before_max**2))
    return None

def process_and_copy_files(correlation_data):
    results = []
    for num_val, data in correlation_data.items():
        target_dir = os.path.join(target_directory, f"grid{num_val}")  
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for index, identifier in enumerate(data['4th_col']):
            folder_name = str(data['3rd_col'][index])
            folder_path = os.path.join(base_directory, folder_name)
            if not os.path.exists(folder_path):
                continue

            files_to_copy = [f"{identifier}.r", f"{identifier}.z"]
            for file in files_to_copy:
                src = os.path.join(folder_path, file)
                dst_filename = f"{data['3rd_col'][index]}_{file}"
                dst = os.path.join(target_dir, dst_filename)
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"Copied {file} to {dst}")
            results.append({
                'numerical_value': num_val,
                '1st_column': data['1st_col'][index],
                'folder': folder_name,
                'identifier': identifier,
                'files': files_to_copy
            })
    return results

def process_and_save_files(correlation_data):
    for num_val, data in correlation_data.items():
        target_dir = os.path.join(target_directory, f"grid{num_val}")  
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        content = []
        for index, identifier in enumerate(data['4th_col']):
            event_id = str(data['3rd_col'][index])  
            folder_name = event_id  
            folder_path = os.path.join(base_directory, folder_name)
            shift_file_path = os.path.join(folder_path, 'shift.dat')

            s_arrival_value = 'N/A'  
            if os.path.exists(shift_file_path):
                with open(shift_file_path, 'r') as shift_file:
                    lines = shift_file.readlines()
                    for line in lines:
                        parts = line.split()
                        if len(parts) > 1 and parts[0].replace(".z", "") == identifier:
                            s_arrival_value = parts[1]  
                            break

            file_path = os.path.join(folder_path, f"{identifier}.z")
            if os.path.exists(file_path):
                st = read(file_path)
                tr = st[0]
                user0 = f"{tr.stats.sac.user0:.3f}" if 'user0' in tr.stats.sac else 'N/A'
                
                sampling_rate = tr.stats.sampling_rate
                waveform_data = tr.data
                rms_value = calculate_rms_noise(waveform_data, sampling_rate)

                rms_value_str = f"{rms_value:.3f}" if rms_value is not None else 'N/A'
                custom_name = f"{folder_name}_{identifier}"
                result_line = f'{custom_name} {user0} {s_arrival_value} {rms_value_str}'
                content.append(result_line)

        with open(os.path.join(target_dir, 'obs.tw'), 'w') as outfile:
            outfile.write('\n'.join(content) + '\n')

def tw_process():
    numerical_values = []
    for folder_name in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, folder_name)):
            num_val = extract_numerical_value(folder_name)
            if num_val is not None:
                numerical_values.append(num_val)

    numerical_values.sort()
    first_column_data, third_column_data, fourth_column_data = read_columns(file_path)

    correlation_data = {}
    for num_val in numerical_values:
        correlated_values = {'1st_col': [], '3rd_col': [], '4th_col': []}
        for i, value in enumerate(first_column_data):
            if abs(value - num_val) <= search_range:
                correlated_values['1st_col'].append(value)
                correlated_values['3rd_col'].append(third_column_data[i])
                correlated_values['4th_col'].append(fourth_column_data[i])
        if correlated_values['1st_col']:  
            correlation_data[num_val] = correlated_values

    process_and_copy_files(correlation_data)
    process_and_save_files(correlation_data)

def tw_obs():
    numerical_values = []
    for folder_name in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, folder_name)):
            num_val = extract_numerical_value(folder_name)
            if num_val is not None:
                numerical_values.append(num_val)

    numerical_values.sort()
    first_column_data, third_column_data, fourth_column_data = read_columns(file_path)

    correlation_data = {}
    for num_val in numerical_values:
        correlated_values = {'1st_col': [], '3rd_col': [], '4th_col': []}
        for i, value in enumerate(first_column_data):
            if abs(value - num_val) <= search_range:
                correlated_values['1st_col'].append(value)
                correlated_values['3rd_col'].append(third_column_data[i])
                correlated_values['4th_col'].append(fourth_column_data[i])
        if correlated_values['1st_col']:  
            correlation_data[num_val] = correlated_values

    process_and_save_files(correlation_data)

def main():
    parser = argparse.ArgumentParser(description='Run the individual Python processing functions')
    parser.add_argument('function', choices=['tw_process', 'tw_obs', 'tw'], help='Function to execute')

    args = parser.parse_args()

    if args.function == 'tw_process':
        tw_process()
    elif args.function == 'tw_obs':
        tw_obs()
    elif args.function == 'tw':
        tw_process()
        tw_obs()

if __name__ == "__main__":
    main()

