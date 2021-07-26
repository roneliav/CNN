import numpy as np
import pandas as pd
from math import sqrt
import time
import os
from scipy import signal
import sys
import ast

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=sys.maxsize)
# np.random.seed(42)
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

def get_data_and_target(path):
    data = pd.read_csv(path, header=None)  # header=None means that there are not columns' names in the csv
    # divide to train and target data frames
    target = data.loc[:, 0].astype(int)  # first column
    data = data.drop(columns=0)  # drop the first column
    data = data.rename(columns=lambda c: c - 1).to_numpy()
    return data, target

def get_weights_from_train(test_folder, epoch_number=0):
    weights = {}
    con_weights_df = pd.read_csv(f"{test_folder}\\epoch_{epoch_number}_con_weights.csv", index_col=0).values.tolist()
    weights['con_weights'] = []
    for i in range(len(con_weights_df)):
    # first_weights = ast.literal_eval(con_weights[0][0])
        level_weights = con_weights_df[i][0]
        level_weights = level_weights.replace('\n','')
        level_weights = level_weights.replace('          ', ',')
        level_weights = level_weights.replace('         ', ',')
        level_weights = level_weights.replace('        ', ',')
        level_weights = level_weights.replace('       ', ',')
        level_weights = level_weights.replace('      ', ',')
        level_weights = level_weights.replace('     ', ',')
        level_weights = level_weights.replace('    ', ',')
        level_weights = level_weights.replace('   ', ',')
        level_weights = level_weights.replace('  ', ',')
        level_weights = level_weights.replace(' -', ',-')
        level_weights = level_weights.replace('] [', '],[')
        level_weights = level_weights.replace(' ', '')
        level_weights = ast.literal_eval(level_weights)
        weights['con_weights'].append(np.asarray(level_weights))
    fully_weights_df = pd.read_csv(f"{test_folder}\\epoch_{epoch_number}_fully_weights.csv", index_col=0).values.tolist()
    weights['fully_weights'] = []
    for i in range(len(fully_weights_df)):
        # first_weights = ast.literal_eval(con_weights[0][0])
        level_weights = fully_weights_df[i][0]
        level_weights = level_weights.replace('\n', '')
        level_weights = level_weights.replace('          ', ',')
        level_weights = level_weights.replace('         ', ',')
        level_weights = level_weights.replace('        ', ',')
        level_weights = level_weights.replace('       ', ',')
        level_weights = level_weights.replace('      ', ',')
        level_weights = level_weights.replace('     ', ',')
        level_weights = level_weights.replace('    ', ',')
        level_weights = level_weights.replace('   ', ',')
        level_weights = level_weights.replace('  ', ',')
        level_weights = level_weights.replace(' -', ',-')
        level_weights = level_weights.replace('] [', '],[')
        level_weights = level_weights.replace(' ', '')
        level_weights = ast.literal_eval(level_weights)
        weights['fully_weights'].append(np.asarray(level_weights))
    return weights


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
    # return (sqrt(6) / sqrt(first_layer_size*9 + second_layer_size*level_features_map_number)) * \
    #        np.random.randn(next_level_feature_maps_number, level_features_map_number, kernel_size, kernel_size)

    # He-Noramlize random
    return sqrt(2 / (first_layer_size*9*level_features_map_number)) * \
               np.random.randn(next_level_feature_maps_number, level_features_map_number, kernel_size, kernel_size)

def print_output_str(test_folder, epoch_number, correct_predict, num_of_rows, lr=False, validate=False, multi_validate=False):
	if multi_validate:
		output_str = f"in epoch_{epoch_number} the validation accuracy precents are {(correct_predict / num_of_rows) * 100}%\n"
		with open(f"{test_folder}\\multi_validate.txt", "a") as output_file:
			output_file.write(output_str)
	elif validate:
		output_str = f"in epoch_{epoch_number} the validation accuracy precents are {(correct_predict / num_of_rows) * 100}%\n"
		with open(f"{test_folder}\\validate.txt", "a") as output_file:
			output_file.write(output_str)
	else:
		output_str = f"in epoch_{epoch_number} the accuracy precents are {(correct_predict / num_of_rows) * 100}%, with lr: {lr}\n"
		with open(f"{test_folder}\\output_layer.txt", "a") as output_file:
			output_file.write(output_str)

	print(output_str)


def from_dict_to_list(weights):
    return [weights[i] for i in range(len(weights))]

def write_weights_to_csv(weights, test_folder, epoch_number):
    pd.DataFrame(data=weights['con_weights']).to_csv(f"{test_folder}\\epoch_{epoch_number}_con_weights.csv")
    weights['fully_weights'] = from_dict_to_list(weights['fully_weights'])
    pd.DataFrame(data=weights['fully_weights']).to_csv(f"{test_folder}\\epoch_{epoch_number}_fully_weights.csv")

def get_predicted_result_from_output_layer(output_layer):
    try:
        return np.where(output_layer == output_layer.max())[0][0] + 1
    except:
        print("dd")
        print(output_layer)

def softmax_der(layer):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    dev = layer.reshape(-1,1)
    return (np.diagflat(dev) - np.dot(dev, dev.T)).diagonal()

def get_error_output(output_layer, target_of_this_row):
    # output_layer_after_der = softmax_der(output_layer)
    target_layer = np.zeros(output_layer.shape)
    target_layer[target_of_this_row - 1] = 1
    error_output = target_layer - output_layer
    # error_output = (target_layer - output_layer) * output_layer_after_der
    return error_output

def get_weights_to_convolution_level(architecture, level):
    level_features_map_number, first_layer_size_rows_size, first_layer_size_columns_size = get_level_shape_from_architecture(architecture, level, return_size=0)
    next_level_feature_maps_number, second_layer_size_rows_size, second_layer_size_columns_size = get_level_shape_from_architecture(architecture, level+1, return_size=0)
    kernel_size = architecture[level]['convolution']['kernel']
    first_layer_size = first_layer_size_rows_size * first_layer_size_columns_size
    second_layer_size = second_layer_size_rows_size * second_layer_size_columns_size
    return get_random_convolution_weights(level_features_map_number, next_level_feature_maps_number, first_layer_size, second_layer_size, kernel_size)

def get_random_weights(architecture):
    weights = {}
    weights['con_weights'] = []
    for level in range(len(architecture)-1): # initialize wieghts for convolutional
        # matrix_size = architecture[level]['convolution']['kernel']
        # np.append(weights['con_weights'], get_weights_to_convolution_level(architecture, level), axis=0)
        weights['con_weights'].append(get_weights_to_convolution_level(architecture, level))
    weights['con_weights'] = np.array(weights['con_weights'])
    weights['fully_weights'] = []
    row_size = get_level_shape_from_architecture(architecture, len(architecture)-1, return_size=1) + 1 # one more for bias
    for level in range(len(architecture['flatten'])-1):
        col_size = architecture['flatten'][level] + 1 # one more for bias
        weights['fully_weights'].append(get_random_matrix(row_size, col_size))
        row_size = col_size
    weights['fully_weights'].append(get_random_matrix(row_size, architecture['flatten'][-1])) # output layer without bias
    weights['fully_weights'] = np.array(weights['fully_weights'])
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
        # num_features_map = architecture[level]['convolution']['features_map']
        fetaure_map_rows_size = after_convolution_row_size / architecture[level]['max_pooling']['kernel']
        fetaure_map_columns_size = after_convolution_col_size / architecture[level]['max_pooling']['kernel']
    return int(after_convolution_row_size)

def max_pooling_one_feature(feature_map, size_of_new_feature_map):
    # architecture = {'kernel':2, 'stride':2}
    # todo: add architecture options not only for 2*2 and stride=2
    new_feature_map = np.empty((size_of_new_feature_map, size_of_new_feature_map))
    for i in range(size_of_new_feature_map):
        for j in range(size_of_new_feature_map):
            new_feature_map[i][j] = feature_map[i*2:i*2+2, j*2:j*2+2].max()
    return new_feature_map


def convolutional_forward_propagation(architecture, input_features, weights, batch_normalization):
    # layers = list, each item is ndarray in shape according architecture
    layers = [None]*(len(architecture))
    layers[0] = {'max_pooling': input_features} # for bacward propagation
    last_layer = input_features
    for level in range(1, len(architecture)): # loop on convolution layers until fully connected layer
        layers[level] = {}
        con_layer_size = architecture[level-1]['convolution']['features_map'] # num of features map after convolution
        size = get_convolution_feature_map_size_in_level(architecture, level-1) # size of each feature map after the convolition
        # convolution
        con_layer = np.zeros((con_layer_size, size, size))
        for feature_next_layer_number in range(con_layer_size):
            for feature_this_layer_number in range(len(last_layer)):
                con_layer[feature_next_layer_number] += signal.correlate2d(last_layer[feature_this_layer_number], weights[level-1][feature_next_layer_number][feature_this_layer_number], mode='same')
        con_layer = np.maximum(con_layer, 0) # ReLU
        if batch_normalization:
            con_layer = normalize_layer(con_layer) # batch normalize
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

def get_dropout_layer(layer, dropout_percent):
    num_of_zeros = int(layer.shape[0] * dropout_percent)
    dropout_vector = np.ones(layer.shape)
    dropout_vector[:num_of_zeros] = 0
    np.random.shuffle(dropout_vector)
    return dropout_vector * layer

def fully_connected_forward_propagation_one_level(layer, weights, dropout=None, validate=False, last=False):
    # if validate and dropout != None:
    #     weights = weights * (1-dropout)
    next_layer = weights * layer[:, np.newaxis]
    next_layer = next_layer.sum(axis=0)
    if not last:
        next_layer = np.maximum(next_layer, 0) # ReLU

        # if dropout != None and not validate:
        #     next_layer = get_dropout_layer(next_layer, dropout)
        next_layer[next_layer.shape[0] - 1] = -1  # bias unit
    return next_layer

def softmax(layer):
    return np.exp(layer) / np.sum(np.exp(layer), axis=0)

def fully_connected_forward_propagation(input_layer, weights, dropout, validate):
    # fully_connected_architecture = [100, 10]
    # dropout = [0.25, 0.45, None]
    num_of_weights = len(weights)
    layers = {}
    layer_number = 0
    # if not validate:
    #     input_layer = get_dropout_layer(input_layer, dropout[0])
    layers[layer_number] = input_layer  # one raw in csv
    # if validate:
    #     dropout.insert(0, None) # the list will be one step forward for dropouting the weights and not the kayer before
    # else:
    #     dropout.append(None) # one more element for last iteration in the for loop
    # if noise_percent:
    #     layers[layer_number] = make_noise(layers[layer_number], noise_percent)
    layers[layer_number] = np.append(layers[layer_number], [-1])  # bias unit
    for layer_number in range(1, num_of_weights + 1):
        layers[layer_number] = fully_connected_forward_propagation_one_level(layers[layer_number-1], weights[layer_number-1], dropout=dropout[layer_number], validate=validate, last=layer_number == num_of_weights)
    layers[layer_number] = softmax(layers[layer_number])
    return layers, layers[layer_number]

def full_forward_propagation(architecture, input_features, weights, batch_normalization, dropout, validate):
    # weights = {'con_weights':.... ,  'fully_weights': .... }
    layers = {'convolution': convolutional_forward_propagation(architecture, input_features, weights['con_weights'], batch_normalization)}
    fully_connected_layer = get_flatten_from_convolution_layers(layers['convolution'])
    # fully_connected_layer = normalize_input_layer(fully_connected_layer)
    layers['fully_connected'], output_layer = fully_connected_forward_propagation(fully_connected_layer, weights['fully_weights'], dropout, validate)
    return layers, output_layer

def get_output_layer_from_layers(layers):
    return layers['fully_connected'][len(layers)]

def backpropagation_one_weights(old_weights, layer, above_layer_error, lr):
    above_layer_rows = above_layer_error.shape[0]
    layer_n = layer
    layer_rows = layer_n.shape[0]
    # new_weights = old_weights + (layer_n.reshape(layer_rows, 1) * above_layer_error.reshape(1, above_layer_rows) * lr)
    new_weights = (layer_n.reshape(layer_rows, 1) * above_layer_error.reshape(1, above_layer_rows) * lr)
    error_layer = ((layer > 0) *1).reshape(layer_rows, 1) * ((old_weights * above_layer_error.reshape(1, above_layer_rows)).sum(axis=1)).reshape(layer_rows, 1)
    return new_weights, error_layer

def fully_connected_backward_propagation(weights, error_output, layers, lr):
    num_of_weights = len(weights)
    updated_weights = [None]*len(weights)
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
    previous_layer_error = np.zeros(previous_layer.shape)
    new_weights = np.empty(old_weights.shape)
    for previous_feature_map_number in range(previous_layer.shape[0]):
        for feature_last_layer_number in range(len(last_layer_error)):
            temp = signal.correlate2d(last_layer_error[feature_last_layer_number], old_weights[feature_last_layer_number][previous_feature_map_number], mode='same')
            previous_layer_error[previous_feature_map_number] += ((previous_layer[previous_feature_map_number]>0)*temp)
    for last_layer_map_number in range(len(last_layer_error)):
        shape = last_layer_error[last_layer_map_number].shape
        last_layer_error_according_weights = np.zeros((9,shape[0], shape[1]))
        last_layer_error_according_weights[0][:-1,:-1] = last_layer_error[last_layer_map_number][1:,1:]
        last_layer_error_according_weights[1][:-1,:] = last_layer_error[last_layer_map_number][1:,:]
        last_layer_error_according_weights[2][:-1, 1:] = last_layer_error[last_layer_map_number][1:, :-1]
        last_layer_error_according_weights[3][:,:-1] = last_layer_error[last_layer_map_number][:,1:]
        last_layer_error_according_weights[4] = last_layer_error[last_layer_map_number]
        last_layer_error_according_weights[5][:, 1:] = last_layer_error[last_layer_map_number][:, :-1]
        last_layer_error_according_weights[6][1:, :-1] = last_layer_error[last_layer_map_number][:-1, 1:]
        last_layer_error_according_weights[7][1:, :] = last_layer_error[last_layer_map_number][:-1, :]
        last_layer_error_according_weights[8][1:, 1:] = last_layer_error[last_layer_map_number][:-1, :-1]
        multiple_errors_by_previous_layer = previous_layer[None,:,:,:]*last_layer_error_according_weights[:,None,:,:]
        sum_errors_of_each_featre_map = multiple_errors_by_previous_layer.sum(axis=(2,3))
        delta_weights = sum_errors_of_each_featre_map.transpose(1,0).reshape((previous_layer.shape[0],3,3))
        # new_weights[last_layer_map_number] = old_weights[last_layer_map_number] + lr * delta_weights
        new_weights[last_layer_map_number] = lr * delta_weights
    return new_weights, previous_layer_error


def convolutional_backward_propagation(weights, output_layer_error, layers, lr):
    last_layer_error = output_layer_error
    new_weights = [None]*len(weights)
    for i in range(len(weights), 0, -1):
        convolution_layer_error = max_pooling_backward_propagation(last_layer_error, layers[i]['convolution'])
        new_weights[i-1], layer_error = specific_convolution_backward_propagation(weights[i-1], convolution_layer_error, layers[i-1]['max_pooling'], lr)
        last_layer_error = layer_error
    return new_weights

def full_backward_propagation(weights, error_output, layers, lr):
    new_weights = {}
    new_weights['fully_weights'], input_layer_error = fully_connected_backward_propagation(weights['fully_weights'], error_output, layers['fully_connected'], lr['fully_connected'])
    last_features_map_layer = input_layer_error[:-1].reshape(layers['convolution'][-1]['max_pooling'].shape) # reshape flatten (except bias unit) to convolution shape
    new_weights['con_weights'] = convolutional_backward_propagation(weights['con_weights'], last_features_map_layer, layers['convolution'], lr['convolution'])
    return new_weights

def normalize_layer(features):
    for i in range(len(features)):
        if features[i].std() != 0:
            features[i] = (features[i] - features[i].mean()) / features[i].std()
        else:
            features[i] = 0

    # for i in range(len(features)):
    #     try:
    #         norm = np.linalg.norm(features[i])
    #         features[i] = features[i] / norm
    #     except:
    #         print("ll")

    return features
    # return (input_features-input_features.mean()) / input_features.std()

def create_rotated_data(train_path, augmented_data_path):
    # write regular rows
    train = pd.read_csv(train_path, header=None).to_numpy()  # header=None means that there are not columns' names in the csv
    for idx, row in enumerate(train):
        with open(augmented_data_path, 'a') as file:
            row = pd.DataFrame(data=row.reshape(1, 3073))
            row.to_csv(augmented_data_path, header=False, index=False, mode='a')

    train = pd.read_csv(train_path, header=None)  # header=None means that there are not columns' names in the csv
    # divide to train and target data frames
    target = train.loc[:, 0]  # first column
    train = train.drop(columns=0)  # drop the first column
    train = train.rename(columns=lambda c: c - 1).to_numpy()

    """"
    write more 7 times: 3 rotate + flip + 3 rotate
    """
    # for k in range(1, 4): # rotate 3 times
    #     for idx, row in enumerate(train):
    #         with open(augmented_data_path, 'a') as file:
    #             file.write(f"{target[idx]},")
    #         rotated_layer = row.reshape(3, 32, 32)
    #         rotated_layer = pd.DataFrame(data=np.rot90(rotated_layer, k=k, axes=(1, 2)).reshape(1,3072))
    #         rotated_layer.to_csv(augmented_data_path, header=False, index=False, mode='a')
    #     print(f"end {k} rotate")
    # for k in range(0, 4):  # rotate 4 times after flip
    #     for idx, row in enumerate(train): # flip right
    #         with open(augmented_data_path, 'a') as file:
    #             file.write(f"{target[idx]},")
    #         layer = row.reshape(3, 32, 32)
    #         flipped_and_rotated = np.rot90(np.flip(layer, 2), k=k, axes=(1,2)).reshape(1, 3072)
    #         rotated_layer =  pd.DataFrame(data=flipped_and_rotated)
    #         rotated_layer.to_csv(augmented_data_path, header=False, index=False, mode='a')

    """
    write flipped rows
    """
    for idx, row in enumerate(train): # flip right - mirror
        with open(augmented_data_path, 'a') as file:
            file.write(f"{target[idx]},")
        layer = row.reshape(3, 32, 32)
        flipped_and_rotated = np.flip(layer, 2).reshape(1, 3072)
        rotated_layer = pd.DataFrame(data=flipped_and_rotated)
        rotated_layer.to_csv(augmented_data_path, header=False, index=False, mode='a')

def add_weights(basic_weights, delta_weights):
    if basic_weights == None:
        return delta_weights.copy()
    basic_weights['con_weights'] = np.array(basic_weights['con_weights']) + np.array(delta_weights['con_weights'])
    basic_weights['fully_weights'] = np.array(basic_weights['fully_weights']) + np.array(delta_weights['fully_weights'])
    # for level in range(len(basic_weights['con_weights'])):
    #     basic_weights['con_weights'][level] = np.array(basic_weights['con_weights'][level]) + np.array(delta_weights['con_weights'][level])
    # for level in range(len(basic_weights['fully_weights'])):
    #     basic_weights['fully_weights'][level] = np.array(basic_weights['fully_weights'][level]) + np.array(delta_weights['fully_weights'][level])
    # assert(basic_weights['con_weights'][0].shape == delta_weights['con_weights'][0].shape)
    # assert(basic_weights['fully_weights'][0].shape == delta_weights['fully_weights'][0].shape)
    return basic_weights.copy()

def get_begin_info(test_folder, architecture, epoch_number, validate, multi_validate):
    if multi_validate:
        if epoch_number == None:
            epoch_number = 0
        mv = 1
        weights = get_weights_from_train(test_folder, epoch_number=f"{epoch_number}_{mv}")
        data, target = get_data_and_target(VALIDATE_PATH)
    elif validate:
        if epoch_number == None:
            epoch_number = 0
        weights = get_weights_from_train(test_folder, epoch_number)
        data, target = get_data_and_target(VALIDATE_PATH)
    else:
        if epoch_number == None:
            weights = get_random_weights(architecture)
            write_weights_to_csv(weights, test_folder, -1)
            data, target = get_data_and_target(TRAIN_PATH)
            epoch_number = 0
        else:
            weights = get_weights_from_train(test_folder, epoch_number)
            epoch_number += 1
            data, target = get_data_and_target(TRAIN_PATH)
    return data, target, weights, epoch_number


def train_convulational_nn(test_folder, architecture, dropout, lr=None, validate=False, normalize=False, epoch_number=None, multi_validate=False, batch_normalization=False, batch_size=None):
    data, target, weights, epoch_number = get_begin_info(test_folder, architecture, epoch_number, validate, multi_validate)
    input_list = get_features_maps_list_from_data(data)

    """
    start training
    """
    while True:  # one epoch for each loop
        correct_predict = 0
        delta_weights = None
        start_time = time.time()
        for i in range(len(target)):
            row_number = i
            input_features = input_list[row_number]
            if normalize:
                input_features = normalize_layer(input_features)
            target_of_this_raw = target[row_number]
            # forward propagation
            layers, output_layer = full_forward_propagation(architecture, input_features, weights, batch_normalization, dropout, validate=(validate or multi_validate))

            predicted_result = get_predicted_result_from_output_layer(output_layer)

            # print and check output
            print(f"excepted: {target_of_this_raw}, got {predicted_result}")
            if target_of_this_raw == predicted_result:
                correct_predict += 1

            if not validate and not multi_validate:  # calculate output error
                if (row_number == 1001):
                    print("y")
                error_output = get_error_output(output_layer, target_of_this_raw)
                # backward propagation
                tmp = full_backward_propagation(weights, error_output, layers, lr)
                # write_weights_to_csv(tmp, test_folder, f"{epoch_number}_{row_number}")
                delta_weights = add_weights(delta_weights, tmp)
                if (row_number + 1) % batch_size == 0:
                    print("update weights")
                    weights = add_weights(weights, delta_weights)
                    delta_weights = None
            print(f"row {i},  {time.time() - start_time} second\n")

            if (((row_number + 1) % 8000) == 0) and (not validate) and (not multi_validate):  # write accuracy precents and write weights to csvs
                write_weights_to_csv(weights, test_folder, f"{epoch_number}_{int((row_number + 1) / 8000)}")
                print_output_str(test_folder, epoch_number, correct_predict, row_number + 1, lr=lr)

        # after full epoch
        if multi_validate:
            print_output_str(test_folder, f"{epoch_number}_{mv}", correct_predict, len(target), multi_validate=True)
            mv += 1
            if mv > 2:
                mv = 1
                epoch_number += 1
            weights = get_weights_from_train(test_folder, epoch_number=f"{epoch_number}_{mv}")
        elif validate:
            print_output_str(test_folder, epoch_number, correct_predict, len(target), validate=True)
            weights = get_weights_from_train(test_folder, epoch_number=epoch_number + 1)
            epoch_number = epoch_number + 1
        else:  # write accuracy precents and write weights to csvs
            write_weights_to_csv(weights, test_folder, epoch_number)
            print_output_str(test_folder, epoch_number, correct_predict, len(target), lr=lr)
            epoch_number = epoch_number + 1
            lr['convolution'] = 0.95 *  lr['convolution']
            lr['fully_connected'] = 0.95 *  lr['fully_connected']


architecture = {0: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 50},
                    'max_pooling': {'kernel':2, 'stride':2}},
                1: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 75},
                    'max_pooling': {'kernel':2, 'stride':2}},
                2: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 125},
                    'max_pooling': {'kernel':2, 'stride':2}},
                # 3: {'convolution': {'padding':1, 'kernel':3, 'stride':1, 'func':'ReLU', 'features_map': 128},
                #     'max_pooling': {'kernel':2, 'stride':2}},
                'flatten': [500, 250, 10]
                }

dropout = [0.25, 0.4, 0.3, None]
# dropout = [0, 0, 0, None]

TRAIN_PATH = "data\\train.csv"
VALIDATE_PATH = "data\\validate.csv"
test_folder = "tt"
lr = {"convolution": 0.01,
      "fully_connected": 0.01}

# create_rotated_data("data\\train.csv", "data\\normal_and_mirror_data.csv")
# os.mkdir(test_folder)
train_convulational_nn(test_folder, architecture, normalize=True, lr=lr, batch_size=50, dropout=dropout)
# train_convulational_nn(test_folder, architecture, normalize=True, validate=True, dropout=dropout, epoch_number=23)
# train_convulational_nn(test_folder, architecture, normalize=True, multi_validate=True, epoch_number=2)

# a = {'con_weights':{0:[1,2], 1:[1,2]}, 'fully_weights':{0:[1,2], 1:[1,2]}}
# b = {'con_weights':{0:[1,1], 1:[1,1]}, 'fully_weights':{0:[1,1], 1:[1,1]}}
# print(a)
# print(b)
# print(add_weights(a,b))