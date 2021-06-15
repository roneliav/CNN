import numpy as np
import pandas as pd
from math import sqrt
import time

"""
archtitecture: a doctionaty
architecture = {0: [{'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 16},
                    'max_pooling': {'kernel':2, 'stride':2}],
                1: [{'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 32},
                    'max_pooling': {'kernel':2, 'stride':2}],
                'flatten': [100],
                }

weights = {'con_weights':
            'fully_weights':
        }

"""

def get_train_and_target():
    train_data = pd.read_csv(TRAIN_PATH, header=None)  # header=None means that there are not columns' names in the csv
    # divide to train and target data frames
    target = train_data.loc[:, 0]  # first column
    train_data = train_data.drop(columns=0)  # drop the first column
    train_data = train_data.rename(columns=lambda c: c - 1).to_numpy()
    return train_data, target

def get_input_size_from_architecture(architecture):
    num_features_map = 3 #input
    fetaure_map_rows_size = 32
    fetaure_map_columns_size = 32
    for level in range(len(architecture)-1):
        kernel = architecture[level]['convolution']['kernel']
        should_padd = int(kernel/2)
        should_padd = should_padd - architecture[level]['convolution']['padding']
        after_convolution_row_size = fetaure_map_rows_size - should_padd
        after_convolution_col_size = fetaure_map_columns_size - should_padd
        num_features_map = architecture[level]['convolution']['features_map']
        fetaure_map_rows_size = after_convolution_row_size / architecture[level]['max_pooling']['kernel']
        fetaure_map_columns_size = after_convolution_col_size / architecture[level]['max_pooling']['kernel']
    return int(num_features_map*fetaure_map_rows_size*fetaure_map_columns_size)

def get_random_matrix(row_size, col_size):
    return sqrt(2 / (row_size + col_size)) * np.random.randn(row_size, col_size)

def get_random_weights(architecture):
    weights = {}
    weights['con_weights'] = []
    for level in range(len(architecture)-1): # initialize wieghts for convolutional
        matrix_size = architecture[level]['convolution']['kernel']
        weights['con_weights'].append(get_random_matrix(matrix_size, matrix_size))
    weights['fully_weights'] = []
    row_size = get_input_size_from_architecture(architecture)
    for level in range(len(architecture['flatten'])):
        col_size = architecture['flatten'][level] + 1 # one more for bias
        weights['fully_weights'].append(get_random_matrix(row_size, col_size))
        row_size = col_size
    weights['fully_weights'].append(get_random_matrix(row_size, 10)) # output layer without bias
    return weights

def get_features_maps_list_from_data(data):
    # divide each row to 3 seperate photos
    data_list = []
    for row in data:
        data_list.append([row[0:1024].reshape(32,32), row[1024:2048].reshape(32,32), row[2048:].reshape(32,32)])
    return data_list

def get_convolution_feature_map_size_in_level(architecture, level):
    fetaure_map_rows_size = 32
    fetaure_map_columns_size = 32
    for lev in range(level+1):
        kernel = architecture[level]['convolution']['kernel']
        should_padd = int(kernel / 2)
        should_padd = should_padd - architecture[level]['convolution']['padding']
        after_convolution_row_size = fetaure_map_rows_size - should_padd
        after_convolution_col_size = fetaure_map_columns_size - should_padd
        if lev == level:
            break
        num_features_map = architecture[level]['convolution']['features_map']
        fetaure_map_rows_size = after_convolution_row_size / architecture[level]['max_pooling']['kernel']
        fetaure_map_columns_size = after_convolution_col_size / architecture[level]['max_pooling']['kernel']
    return int(after_convolution_row_size)

def do_convolution_to_one_entry(one_feature_in_last_layer, row, col, weights_matrix, architecture):
    # architecture = {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 16}
    kernel_size = architecture['kernel']
    # should_padd =
    half_kernel = int(kernel_size/2)
    # if row < half_kernel:
        

    feature_map = np.zeros(one_feature_in_last_layer.shape)
    subset_mat = last_layer_map[row-add_subset:row+add_subset, col-add_subset:col-add_subset]
    feature_map[row][col] += subset_mat*weights_matrix
    return entry

def convolutional_forward_propagation(architecture, input_features, weights):
    layers = [None]*(len(architecture)-1)
    last_layer = input_features
    for level in range(len(architecture)-1):
        layers[level] = {}
        con_layer_size = architecture[level]['convolution']['features_map'] #num of features map after concolution
        size = get_convolution_feature_map_size_in_level(architecture, level) #size of each feature map after the convolition
        # convolution
        con_layer = np.zeros((con_layer_size, size, size))
        for feature_next_layer_number in con_layer_size:
            feature_map = con_layer[feature_next_layer_number]
            for map in range(last_layer):
                for row in size:
                    for col in size:
                        feature_map[row][col] = feature_map[row][col] + do_convolution_to_one_entry(last_layer[map], row, col, weights[level * 2], architecture[level]['convolution'])
            con_layer[feature_next_layer_number] = feature_map #todo: add ReLU
        layers[level]['convolution'] = con_layer

        # max-pooling
        last_layer = con_layer
        size = size / architecture[level]['max_pooling']['kernel']  # todo: add stride
        max_pooling_layer = np.zeros((con_layer_size, size, size))
        for feature_number in con_layer_size:
            max_pooling_layer[feature_number] = max_pooling_one_feature(con_layer[feature_number], architecture[level]['max_pooling'])
        layers[level]['max_pooling'] = max_pooling_layer
        last_layer = max_pooling_layer
    return layers


def full_forward_propagation(architecture, input_features, weights):
    layers = {}
    layers['convolution'] = convolutional_forward_propagation(architecture, input_features, weights)
    fully_connected_layer = get_fully_connected_from_last_convolution(layers['convolution'])
    layers['fully_connected'] = fully_connected_forward_propagation(architecture['flatten'], fully_connected_layer, weights)
    return layers

def train_convulational_nn(architecture, test_folder, lr):
    weights = get_random_weights(architecture)
    data, target = get_train_and_target()
    input_list = get_features_maps_list_from_data(data)
    epoch_number = 0
    """
    start training
    """
    while (True):  # one epoch for each loop
        correct_predict = 0
        start_time = time.time()
        for i in range(len(target)):
            row_number = i
            input_features = input_list[row_number]
            target_of_this_raw = target[row_number]
            # forward propagation
            layers = full_forward_propagation(architecture, input_features, weights)
            output_layer = get_output_layer_from_layers(layers)
            predicted_result = get_predicted_result_from_output_layer(output_layer)

            """
            print and check output
            """
            print(f"excepted: {target_of_this_raw}, got {predicted_result}")
            if target_of_this_raw == predicted_result:
                correct_predict += 1

            # calculate output error
            error_output = get_error_output(output_layer, target_of_this_raw)
            # backward propagation
            weights = full_forward_propagation(weights, error_output, layers, lr)
            print(f"row {i},  {time.time() - start_time} second\n")

        # after full epoch - write accuracy precents and write weights to csvs
        print_output_str(test_folder, epoch_number, correct_predict, lr)
        write_weights_to_csv(between_layers_weights, test_folder, epoch_number)
        epoch_number = epoch_number + 1
        # lr = 0.95 * lr



architecture = {0: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 16},
                    'max_pooling': {'kernel':2, 'stride':2}},
                1: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 32},
                    'max_pooling': {'kernel':2, 'stride':2}},
                'flatten': [100],
                }
TRAIN_PATH = "data\\train.csv"
# train_convulational_nn(architecture, "a", 0)
data, target = get_train_and_target()
input_list = get_features_maps_list_from_data(data)
print(len(input_list[0]))
print(input_list[0][0].shape)
print(target[0])
