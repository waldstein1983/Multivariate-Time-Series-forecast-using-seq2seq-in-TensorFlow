import tensorflow as tf
import numpy as np
import random
import math
from matplotlib import pyplot as plt
import os
import copy
from build_model_basic import *
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler

x = np.linspace(0, 30, 105)
train_data_x = x[:85]

y = 2 * np.sin(x)

learning_rate = 0.01
lambda_l2_reg = 0.003


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


series = read_csv('../CorpData/InventoryHistory/2010_2018_books_sortable inventory.csv',
                  header=0, parse_dates=[0], index_col=0, squeeze=True, usecols=[0, 4])
## Network Parameters
# length of input signals
input_seq_len = 15
# length of output signals
output_seq_len = 20

n_test = 15

scaler, train, test = prepare_data(series, n_test, input_seq_len, output_seq_len)
# size of LSTM Cell
hidden_dim = 64
# num of input signals
input_dim = 1
# num of output signals
output_dim = 1
# num of stacked lstm layers
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5

total_iteractions = 100
batch_size = 1
KEEP_RATE = 0.5
train_losses = []
val_losses = []


def true_signal(x):
    y = 2 * np.sin(x)
    return y


def noise_func(x, noise_factor=1):
    return np.random.randn(len(x)) * noise_factor


def generate_y_values(x):
    return true_signal(x) + noise_func(x)


train_X, train_y = train[:, 0:input_seq_len], train[:, input_seq_len:]


def generate_train_samples(x=train_X, batch_size=10, input_seq_len=input_seq_len, output_seq_len=output_seq_len):
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size)

    input_seq_x = [x[i:(i + input_seq_len)] for i in start_x_idx]
    output_seq_x = [x[(i + input_seq_len):(i + input_seq_len + output_seq_len)] for i in start_x_idx]

    input_seq_y = [generate_y_values(x) for x in input_seq_x]
    output_seq_y = [generate_y_values(x) for x in output_seq_x]

    # batch_x = np.array([[true_signal()]])
    return np.array(input_seq_y), np.array(output_seq_y)


x = np.linspace(0, 30, 105)
train_data_x = x[:85]

rnn_model = build_graph(feed_previous=False)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(train_X)):
        batch_input = train_X[i].reshape(1, train_X.shape[1])
        # batch_input = batch_input.reshape(1, batch_input.shape[0])
        batch_output = train_y[i].reshape(1, train_y.shape[1])
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t].reshape(-1, input_dim) for t in range(input_seq_len)}
        feed_dict.update(
            {rnn_model['target_seq'][t]: batch_output[:, t].reshape(-1, output_dim) for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)

    # for i in range(total_iteractions):
    #     batch_input, batch_output = generate_train_samples(batch_size=batch_size)
    #
    #     feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t].reshape(-1, input_dim) for t in range(input_seq_len)}
    #     feed_dict.update(
    #         {rnn_model['target_seq'][t]: batch_output[:, t].reshape(-1, output_dim) for t in range(output_seq_len)})
    #     _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
    #     print(loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'univariate_ts_model0'))

print("Checkpoint saved at: ", save_path)

# test_seq_input = true_signal(train_data_x[-15:])
raw_values = series.values
# transform data to be stationary
diff_series = difference(raw_values, 1)
diff_values = diff_series.values

test_seq_input = diff_values[-15:]
# test_X, test_y = test[:, 0:input_seq_len], test[:, input_seq_len:]

rnn_model = build_graph(feed_previous=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    saver = rnn_model['saver']().restore(sess, os.path.join('./', 'univariate_ts_model0'))

    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1, 1) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

    final_preds = np.concatenate(final_preds, axis=1)

last_observe = raw_values[-1:]


def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        # forecast = array(forecasts[i])
        forecast = forecasts[i].asnumpy();
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# inverted_original = inverse_difference(last_observe, final_preds[0])
# inverted = list()

inv_scale = scaler.inverse_transform(final_preds)
inv_scale = inv_scale[0, :]
inverted = inverse_difference(last_observe, inv_scale)

plt.plot(raw_values)
plt.plot(inverted)
plt.show()
for i in range(len(inverted)):
    print(inverted[i])

# for i in range(len(final_preds[0])):
#     # create array from forecast
#     # forecast = array(forecasts[i])
#     forecast = final_preds[0][i].asnumpy();
#     forecast = forecast.reshape(1, len(forecast))
#     # invert scaling
#     inv_scale = scaler.inverse_transform(forecast)
#     inv_scale = inv_scale[0, :]
#     # invert differencing
#     # index = len(series) - n_test + i - 1
#     # last_ob = series.values[index]
#     inv_diff = inverse_difference(last_observe, inv_scale)
#     # store
#     inverted.append(inv_diff)
#
# for i in range(len(inverted_original)):
#     print (inverted_original[i])

# # l1, = plt.plot(range(85), true_signal(train_data_x[:85]), label = 'Training truth')
# l1, = plt.plot(train_X, label = 'Training truth')
# l2, = plt.plot(train_y, 'yo', label = 'Test truth')
# l3, = plt.plot(range(85, 105), final_preds.reshape(-1), 'ro', label = 'Test predictions')
# plt.legend(handles = [l1, l2, l3], loc = 'lower left')
# plt.show()
