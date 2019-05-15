# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob

TIME_COLUMN = "time (01:00)"



class MtbDataProvider:

    @staticmethod
    def load_data(acc_data_filepath, acc_data_file_ending, frequency):
        # Guess the labels file name
        acc_labels_filepath = acc_data_filepath + '_labels.csv'

        acc_data_filepaths = glob.glob(acc_data_filepath + acc_data_file_ending)
        print(acc_data_filepaths, acc_labels_filepath)

        acc_datas = []
        for acc_data_filepath in acc_data_filepaths:

            # Read CSV files
            acc_data = pd.read_csv(acc_data_filepath)

            # Fill data_processing gaps
            acc_data[TIME_COLUMN] = pd.to_datetime(acc_data[TIME_COLUMN])
            acc_data = acc_data.set_index(TIME_COLUMN)
            acc_data.asfreq(str(frequency) + 'L')

            # Interpolate Data
            for column in acc_data.columns:
                acc_data[column].interpolate(inplace=True, limit_direction='both')

            acc_datas.append(acc_data.to_numpy())

        acc_labels = pd.read_csv(acc_labels_filepath)
        acc_labels = acc_labels.to_numpy()

        # If there was only one file, just return the first value
        if len(acc_data_filepaths) == 1:
            return acc_datas[0], acc_labels
        else:
            return np.asarray(acc_datas), acc_labels


    @staticmethod
    def slice_single_sensor(acc_data, acc_labels, window_size, frequency):

        sample_size = window_size // frequency

        current_label_row_i = 0
        current_label_row = acc_labels[0]
        start_epoch = acc_data[0][0]

        X = []
        y = []

        # Iterate through slices of rows
        for i in range(0, acc_data.shape[0], sample_size):

            # Grab the first slice
            if i + sample_size > acc_data.shape[0]:
                acc_slice = acc_data[-sample_size:]
            else:
                acc_slice = acc_data[i:i + sample_size]

            # Get last timestamp of this slice
            first_acceleration_slice_start = acc_slice[0, 0]
            # Calculate relative time difference to start
            relative_slice_start = first_acceleration_slice_start - start_epoch

            # Get the end-time of the current label
            # * 1000 to create milliseconds
            current_label_end = current_label_row[1] * 1000

            too_small_overlap = (current_label_end - relative_slice_start) / window_size < 0.5
            if too_small_overlap and current_label_row_i < acc_labels.shape[0] - 1:
                current_label_row_i += 1
                current_label_row = acc_labels[current_label_row_i]

            # Append the slice and difficulty
            X.append(acc_slice[:, 2:5])
            y.append(current_label_row[2])

        return np.array(X), np.asarray(y)

    @staticmethod
    def slice_multiple_sensors(acc_datas, acc_labels, window_size, frequency):

        sample_size = window_size // frequency

        current_label_row_i = 0
        current_label_row = acc_labels[0]
        start_epoch = acc_datas[0, 0, 0]

        X = []
        y = []

        # Iterate through slices of rows
        for i in range(0, acc_datas.shape[0], sample_size):

            # Grab the first slice
            if i + sample_size > acc_datas.shape[0]:
                acc_slice = acc_datas[-sample_size:, :, :]
            else:
                acc_slice = acc_datas[i:i + sample_size, :, :]

            # Get last timestamp of this slice
            first_acceleration_slice_start = acc_slice[0, 0, 0]
            # Calculate relative time difference to start
            relative_slice_start = first_acceleration_slice_start - start_epoch

            # Get the end-time of the current label
            # * 1000 to create milliseconds
            current_label_end = current_label_row[1] * 1000

            too_small_overlap = (current_label_end - relative_slice_start) / window_size < 0.5
            if too_small_overlap and current_label_row_i < acc_labels.shape[0] - 1:
                current_label_row_i += 1
                current_label_row = acc_labels[current_label_row_i]

            # Append the slice and difficulty
            X.append(acc_slice[:, :, 2:5])
            y.append(current_label_row[2])

        return np.array(X), np.asarray(y)

    @staticmethod
    def sync_sensors(sensors_data):
        X = []

        # Check what the latest sensor start is (so that all start at this point in time)
        starts = []
        for sensor in sensors_data:
            starts.append(sensor[0, 0])
        latest_sensor_start = max(starts)

        # Crop the data at the beginning, that does not overlap (let all sensors start at the latest_sensor_start)
        lengths = []
        for sensor in sensors_data:
            # If the sensor starts earlier than the latest sensor, remove that data
            sensor_start = sensor[0, 0]
            while sensor_start < latest_sensor_start:
                sensor = sensor[1:]
                sensor_start = sensor[0, 0]

            lengths.append(len(sensor))

        # Crop the data at the end, so that all data is of equal length
        tmp_x = []
        min_length = min(lengths)
        for sensor in sensors_data:
            tmp_x.append(sensor[:min_length])
        tmp_x = np.asarray(tmp_x)

        for i in range(0, min_length):
            X.append(np.vstack(tmp_x[:, i]))

        return np.asarray(X)

