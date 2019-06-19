# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
from sklearn.utils import shuffle

TIME_COLUMN = "time (01:00)"


class MtbDataProvider:

    @staticmethod
    def load_data(data_filepath, data_file_ending, frequency):
        # Guess the labels file name
        labels_filepath = data_filepath + '_labels.csv'

        data_filepaths = sorted(glob.glob(data_filepath + data_file_ending))

        data_array = []
        for data_filepath in data_filepaths:
            print("reading ", data_filepath)

            # Read CSV files
            data = pd.read_csv(data_filepath)

            # Fill data_processing gaps
            data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN])
            data = data.set_index(TIME_COLUMN)
            data = data.asfreq(str(frequency) + 'L')

            # Interpolate Data
            for column in data.columns:
                data[column].interpolate(inplace=True, limit_direction='both')

            data_array.append(data.to_numpy())

        labels = pd.read_csv(labels_filepath)
        labels = labels.to_numpy()

        # If there was only one file, just return the first value
        if len(data_filepaths) == 1:
            return data_array[0], labels
        else:
            return np.asarray(data_array), labels

    @staticmethod
    def slice_sensors(data_array, sensor_labels, window_size, frequency, step_size=0.5, dismiss_not_riding_label=False):

        sample_size = window_size // frequency

        current_label_row_i = 0
        current_label_row = sensor_labels[0]
        start_epoch = data_array[0, 0, 0]

        X = []
        y = []

        # Iterate through slices of rows
        stride = int(sample_size // (1 // step_size))
        for i in range(0, data_array.shape[0], stride):

            # Grab the first slice
            if i + sample_size > data_array.shape[0]:
                slice = data_array[-sample_size:, :, :]
            else:
                slice = data_array[i:i + sample_size, :, :]

            # Get last timestamp of this slice
            first_slice_start = slice[0, 0, 0]
            # Calculate relative time difference to start
            relative_slice_start = first_slice_start - start_epoch

            # Get the end-time of the current label
            # * 1000 to create milliseconds
            current_label_end = current_label_row[1] * 1000

            too_small_overlap = (current_label_end - relative_slice_start) / window_size < 0.5
            if too_small_overlap:
                # If there is a next label, take it
                if current_label_row_i < sensor_labels.shape[0] - 1:
                    current_label_row_i += 1
                    current_label_row = sensor_labels[current_label_row_i]
                else: # if there is no next label: cancel here
                    break

            if dismiss_not_riding_label and current_label_row[2] is 3:
                continue


            # Append the slice and difficulty
            X.append(slice[:, :, 2:5])
            y.append(current_label_row[2])

        return np.array(X), np.asarray(y)

    @staticmethod
    def evenly_oversample(X, y):

        X_result = []
        y_result = []

        unique, counts = np.unique(y, return_counts=True)
        max_count = max(counts)
        counts_dict = dict(zip(unique, counts))
        oversample_count = {
            0: -max_count,
            1: -max_count,
            2: -max_count,
        }

        for i in range(0, X.shape[0]):
            label = y[i]
            multiply_factor = int(np.ceil(max_count / counts_dict[label]))
            for j in range(0, multiply_factor):
                if oversample_count[label] < 0:
                    X_result.append(X[i])
                    y_result.append(y[i])
                    oversample_count[label] += 1

        X_result = np.array(X_result)
        y_result = np.array(y_result)


        X_result, y_result = shuffle(X_result, y_result)

        return X_result, y_result



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
        shortest_length = min(lengths)
        for sensor in sensors_data:
            cropped_sensor = sensor[:shortest_length]

            # In case this is a 1-dimensional sensor, the data has only (:, 3) dimensions
            # Therefore copy the last column 2 times, so we have 3 channels as well
            # TODO: This is a bit hacky
            if cropped_sensor.shape[1] == 3:
                last_row = cropped_sensor[:, 2:]
                cropped_sensor = np.hstack((cropped_sensor, last_row))
                cropped_sensor = np.hstack((cropped_sensor, last_row))
            tmp_x.append(cropped_sensor)

        tmp_x = np.asarray(tmp_x)

        for i in range(0, shortest_length):
            X.append(np.vstack(tmp_x[:, i]))

        return np.asarray(X)
