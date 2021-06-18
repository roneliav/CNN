import numpy as np
import pandas as pd
from math import sqrt
import time
import os
from scipy import signal

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

def get_level_shape_from_architecture(architecture, last_level, return_size=0):
    num_features_map = 3 #input
    fetaure_map_rows_size = 32
    fetaure_map_columns_size = 32
    for level in range(last_level):
        kernel = architecture[level]['convolution']['kernel']
        should_padd = int(kernel/2)
        should_padd = should_padd - architecture[level]['convolution']['padding']
        after_convolution_row_size = fetaure_map_rows_size - should_padd
        after_convolution_col_size = fetaure_map_columns_size - should_padd
        num_features_map = architecture[level]['convolution']['features_map']
        fetaure_map_rows_size = after_convolution_row_size / architecture[level]['max_pooling']['kernel']
        fetaure_map_columns_size = after_convolution_col_size / architecture[level]['max_pooling']['kernel']
    if return_size:
        return int(num_features_map * fetaure_map_rows_size * fetaure_map_columns_size)
    else:
        return num_features_map, fetaure_map_rows_size, fetaure_map_columns_size

def get_random_matrix(row_size, col_size):
    return sqrt(2 / (row_size + col_size)) * np.random.randn(row_size, col_size)

def get_random_convolution_weights(level_features_map_number, next_level_feature_maps_number,first_layer_size, second_layer_size, kernel_size):
    # xavier random:
    # return (sqrt(6) / sqrt((first_layer_size + second_layer_size))) * \
    #        np.random.randn(next_level_feature_maps_number, level_features_map_number, kernel_size, kernel_size)

    # He-Noramlize random
    return sqrt(2 / (first_layer_size + second_layer_size)) * \
               np.random.randn(next_level_feature_maps_number, level_features_map_number, kernel_size, kernel_size)

def print_output_str(test_folder, epoch_number, correct_predict, lr):
    output_str = f"in epoch_{epoch_number} the accuracy precents are {(correct_predict / 8000) * 100}, with {lr} lr%\n"
    print(output_str)
    with open(f"{test_folder}\\output_layer.txt", "a") as output_file:
        output_file.write(output_str)

def from_dict_to_list(weights):
    return [weights[i] for i in range(len(weights))]

def write_weights_to_csv(weights, test_folder, epoch_number):
    pd.DataFrame(data=weights['con_weights']).to_csv(f"{test_folder}\\epoch_{epoch_number}_con_weigts.csv")
    weights['fully_weights'] = from_dict_to_list(weights['fully_weights'])
    pd.DataFrame(data=weights['fully_weights']).to_csv(f"{test_folder}\\epoch_{epoch_number}_fully_weights.csv")

def get_predicted_result_from_output_layer(output_layer):
    return np.where(output_layer == output_layer.max())[0][0] + 1

def get_error_output(output_layer, target_of_this_row):
    target_layer = np.zeros(output_layer.shape)
    target_layer[target_of_this_row - 1] = 1
    error_output = target_layer - output_layer
    return error_output

def get_weights_to_convolution_level(architecture, level):
    level_features_map_number, first_layer_size_rows_size, first_layer_size_columns_size = get_level_shape_from_architecture(architecture, level, return_size=0)
    next_level_feature_maps_number, second_layer_size_rows_size, second_layer_size_columns_size = get_level_shape_from_architecture(architecture, level+1, return_size=0)
    kernel_size = architecture[level]['convolution']['kernel']
    first_layer_size = first_layer_size_rows_size * first_layer_size_columns_size
    second_layer_size = second_layer_size_rows_size * second_layer_size_columns_size
    return get_random_convolution_weights(level_features_map_number, next_level_feature_maps_number,first_layer_size, second_layer_size, kernel_size)

def get_random_weights(architecture):
    weights = {}
    weights['con_weights'] = []
    for level in range(len(architecture)-1): # initialize wieghts for convolutional
        matrix_size = architecture[level]['convolution']['kernel']
        weights['con_weights'].append(get_weights_to_convolution_level(architecture, level))
    weights['fully_weights'] = []
    row_size = get_level_shape_from_architecture(architecture, len(architecture)-1, return_size=1) + 1 # one more for bias
    for level in range(len(architecture['flatten'])-1):
        col_size = architecture['flatten'][level] + 1 # one more for bias
        weights['fully_weights'].append(get_random_matrix(row_size, col_size))
        row_size = col_size
    weights['fully_weights'].append(get_random_matrix(row_size, architecture['flatten'][-1])) # output layer without bias
    return weights

def get_features_maps_list_from_data(data):
    # divide each row to 3 seperate photos
    data_list = []
    for row in data:
        data_list.append(np.array([row[0:1024].reshape(32,32), row[1024:2048].reshape(32,32), row[2048:].reshape(32,32)]))
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

def max_pooling_one_feature(feature_map, size_of_new_feature_map):
    # architecture = {'kernel':2, 'stride':2}
    # todo: add architecture options not only for 2*2 and stride=2
    new_feature_map = np.empty((size_of_new_feature_map, size_of_new_feature_map))
    for i in range(size_of_new_feature_map):
        for j in range(size_of_new_feature_map):
            new_feature_map[i][j] = max(feature_map[2*i,2*j], feature_map[2*i,2*j+1], \
                                        feature_map[2*i+1,2*j], feature_map[2*i+1,2*j+1])
    return new_feature_map


def convolutional_forward_propagation(architecture, input_features, weights):
    # layers = list, each item is ndarray in shape according architecture
    layers = [None]*(len(architecture))
    layers[0] = {'max_pooling': input_features}
    last_layer = input_features
    for level in range(1, len(architecture)): # loop on convolution layers until fully connected layer
        layers[level] = {}
        con_layer_size = architecture[level-1]['convolution']['features_map'] # num of features map after convolution
        size = get_convolution_feature_map_size_in_level(architecture, level-1) # size of each feature map after the convolition
        # convolution
        con_layer = np.zeros((con_layer_size, size, size))
        for feature_next_layer_number in range(con_layer_size):
            con_layer[feature_next_layer_number] = signal.correlate(last_layer, weights[level-1][feature_next_layer_number], mode='same').sum(axis=0)
            # con_layer[feature_next_layer_number] = np.maximum(con_layer[feature_next_layer_number], 0) # ReLU
        con_layer = np.maximum(con_layer, 0) # ReLU
        layers[level]['convolution'] = con_layer

        # max-pooling
        size = int(size / architecture[level-1]['max_pooling']['kernel'])  # todo: add stride
        max_pooling_layer = np.empty((con_layer_size, size, size))
        for feature_number in range(con_layer_size):
            max_pooling_layer[feature_number] = max_pooling_one_feature(con_layer[feature_number], size)
        layers[level]['max_pooling'] = max_pooling_layer
        last_layer = max_pooling_layer
    return layers

def get_flatten_from_convolution_layers(layers):
    return layers[-1]['max_pooling'].flatten()

def fully_connected_forward_propagation_one_level(layer, weights, last=False):
    next_layer = weights * layer[:, np.newaxis]
    next_layer = next_layer.sum(axis=0)
    next_layer = np.maximum(next_layer, 0) # ReLU
    if not last:
        next_layer[next_layer.shape[0] - 1] = -1  # bias unit
    return next_layer

def fully_connected_forward_propagation(input_layer, weights):
    # fully_connected_architecture = [100, 10]
    num_of_weights = len(weights)
    layers = {}
    layer_number = 0
    layers[layer_number] = input_layer  # one raw in csv
    # if noise_percent:
    #     layers[layer_number] = make_noise(layers[layer_number], noise_percent)
    layers[layer_number] = np.append(layers[layer_number], [-1])  # bias unit
    for layer_number in range(1, num_of_weights + 1):
        layers[layer_number] = fully_connected_forward_propagation_one_level(layers[layer_number - 1], weights[layer_number - 1], last=layer_number == num_of_weights)
    return layers, layers[layer_number]

def full_forward_propagation(architecture, input_features, weights):
    # weights = {'con_weights':.... ,  'fully_weights': .... }
    layers = {'convolution': convolutional_forward_propagation(architecture, input_features, weights['con_weights'])}
    fully_connected_layer = get_flatten_from_convolution_layers(layers['convolution'])
    # fully_connected_layer - normalize_input_layer(fully_connected_layer)
    layers['fully_connected'], output_layer = fully_connected_forward_propagation(fully_connected_layer, weights['fully_weights'])
    return layers, output_layer

def get_output_layer_from_layers(layers):
    return layers['fully_connected'][len(layers)]

def backpropagation_one_weights(old_weights, layer, above_layer_error, lr):
    above_layer_rows = above_layer_error.shape[0]
    layer_n = layer
    layer_rows = layer_n.shape[0]
    new_weights = old_weights + (layer_n.reshape(layer_rows, 1) * above_layer_error.reshape(1, above_layer_rows) * lr)
    error_layer = ((layer > 0) *1).reshape(layer_rows, 1) * ((old_weights * above_layer_error.reshape(1, above_layer_rows)).sum(axis=1)).reshape(layer_rows, 1)
    return new_weights, error_layer

def fully_connected_backward_propagation(weights, error_output, layers, lr):
    num_of_weights = len(weights)
    updated_weights = {}
    above_layer_error = error_output
    for layer_number in range(num_of_weights - 1, -1, -1):
        updated_weights[layer_number], above_layer_error = backpropagation_one_weights(weights[layer_number], layers[layer_number], above_layer_error, lr)
    return updated_weights, above_layer_error

def max_pooling_backward_propagation(last_layer_error, previous_layer):
    # return previous_layer error
    # there are the same number of featres maps at last_layer and at previous_layer
    previous_layer_error = np.zeros(previous_layer.shape)
    for map_number in range(len(previous_layer)):
        previous_map = previous_layer[map_number]
        last_layer_error_map = last_layer_error[map_number]
        for row in range(0, last_layer_error_map.shape[0], 1):
            for col in range(0, last_layer_error_map.shape[1], 1):
                max_index = np.where(previous_map[row*2:row*2+2, col*2:col*2+2] == previous_map[row*2:row*2+2, col*2:col*2+2].max())
                previous_layer_error[map_number][row*2+max_index[0][0]][col*2+max_index[1][0]] = last_layer_error_map[row][col]
    return previous_layer_error


def specific_convolution_backward_propagation(old_weights, last_layer_error, previous_layer, lr):
    # return previous_layer error and the updated weights
    old_weights_for_back = old_weights.transpose(1,0,2,3)
    previous_layer_error = np.zeros(previous_layer.shape)
    new_weights = np.empty(old_weights.shape)
    for previous_feature_map_number in range(previous_layer.shape[0]):
        previous_layer_error[previous_feature_map_number] = signal.correlate(last_layer_error, old_weights_for_back[previous_feature_map_number], mode='same').sum(axis=0) # todo: add relu
        previous_layer_error[previous_feature_map_number] = (previous_layer_error[previous_feature_map_number]>0)*1
    for last_layer_map_number in range(len(last_layer_error)):
        for previous_layer_map_number in range(len(previous_layer)):
            current_old_weights = old_weights[last_layer_map_number][previous_layer_map_number]
            multiple_error = (previous_layer[previous_layer_map_number] * last_layer_error[last_layer_map_number]).sum()
            new_weights[last_layer_map_number][previous_layer_map_number] = current_old_weights + current_old_weights * multiple_error * lr
    return new_weights, previous_layer_error


def convolutional_backward_propagation(weights, output_layer_error, layers, lr):
    last_layer_error = output_layer_error
    for i in range(len(weights), 0, -1):
        convolution_layer_error = max_pooling_backward_propagation(last_layer_error, layers[i]['convolution'])
        weights[i-1], layer_error = specific_convolution_backward_propagation(weights[i-1], convolution_layer_error, layers[i-1]['max_pooling'], lr)
        last_layer_error = layer_error
    return weights

def full_backward_propagation(weights, error_output, layers, lr):
    new_weights = {}
    new_weights['fully_weights'], input_layer_error = fully_connected_backward_propagation(weights['fully_weights'], error_output, layers['fully_connected'], lr['fully_connected'])
    last_features_map_layer = input_layer_error[:-1].reshape(layers['convolution'][-1]['max_pooling'].shape) # reshape flatten (except bias unit) to convolution shape
    new_weights['con_weights'] = convolutional_backward_propagation(weights['con_weights'], last_features_map_layer, layers['convolution'], lr['convolution'])
    return new_weights

def normalize_layer(input_features):
    # from sklearn.preprocessing import StandardScaler
    for i in range(len(input_features)):
        # input_features[i] = StandardScaler().fit_transform(input_features[i])
        input_features[i] = (input_features[i]-input_features[i].mean()) / input_features[i].std()
    return input_features
    # return (input_features-input_features.mean()) / input_features.std()

def train_convulational_nn(architecture, test_folder, lr):
    weights = get_random_weights(architecture)
    write_weights_to_csv(weights, test_folder, -1)
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
            # input_features = normalize_layer(input_features)
            target_of_this_raw = target[row_number]
            # forward propagation
            layers, output_layer = full_forward_propagation(architecture, input_features, weights)
            # output_layer = get_output_layer_from_layers(layers)
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
            weights = full_backward_propagation(weights, error_output, layers, lr)
            print(f"row {i},  {time.time() - start_time} second\n")

        # after full epoch - write accuracy precents and write weights to csvs
        print_output_str(test_folder, epoch_number, correct_predict, lr)
        write_weights_to_csv(weights, test_folder, epoch_number)
        epoch_number = epoch_number + 1
        # lr = 0.95 * lr



architecture = {0: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 16},
                    'max_pooling': {'kernel':2, 'stride':2}},
                1: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 32},
                    'max_pooling': {'kernel':2, 'stride':2}},
                'flatten': [100, 10],
                }

TRAIN_PATH = "data\\train.csv"
test_folder = "b"
lr = {"convolution": 0.001,
      "fully_connected": 0.001}
os.mkdir(test_folder)
train_convulational_nn(architecture, test_folder, lr=lr)

# c = np.arange(16).reshape(4, 4)
# d = np.arange(9).reshape(3, 3)
# print(c)
# print(d)
# print(signal.convolve(d, c, mode='same'))
# print(signal.convolve(d, c, mode='full'))
# print(c)
# print(d)
# e = d[:,:,None,None]*c
# print(e)
# g = e.sum(axis=(2, 3))
# print(g)
# print(signal.correlate(last_layer_error, weights, mode='same').sum(axis=0))

