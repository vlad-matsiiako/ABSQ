import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from utils import calculate_classifications


DATASET_NUMBER = 4000
TEST_DATASET_NUMBER = 100
PADDING_SIZE = 60
BATCH_SIZE = 32
NUM_OF_EPOCHS = 3
MIN_SIZE = 0


df_train = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_train_service', delimiter=" ", header=None)
df_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_test_service', delimiter=" ", header=None)
df_categories_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/data/programGeneratedData/categories_test_2016.csv', delimiter=",").drop(columns=['Unnamed: 0'])

df_test = df_test[df_categories_test["Category"] == "RESTAURANT#GENERAL"].reset_index(drop=True)


data = []
quantifications = []
additional_data = []
for i in range(DATASET_NUMBER):
    np.random.seed(i)
    data.append(np.array(df_train.sample(n=PADDING_SIZE-1, replace=True, random_state=i).append(df_train[df_train[0]==2].sample(n=1, random_state=i)).reset_index(drop=True)))
    additional_data.append(calculate_classifications(data[i][:, 0], data[i][:, 1], data[i][:, np.r_[2:5]]))
    length = len(data[i])
    quantifications.append([np.count_nonzero(data[i][:, 0] == 0) / length, np.count_nonzero(data[i][:, 0] == 1) / length, np.count_nonzero(data[i][:, 0] == 2) / length])
    data[i] = data[i][:, np.r_[2:2405]]
    data[i] = np.c_[data[i], - data[i][:, 0]*np.log(data[i][:, 0]) - data[i][:, 1]*np.log(data[i][:, 1]) - data[i][:, 2]*np.log(data[i][:, 2])]
    # Choose depending on what you want to sort: 0 - probability, 2403 - entropy
    data[i] = data[i][data[i][:, 2403].argsort()]
    # data[i] = data[i][data[i][:, 0].argsort()]
    print(i)


data = np.asarray(data)
additional_data = np.asarray(additional_data)
quantifications = np.asarray(quantifications)


test_data = []
test_quantifications = []
test_additional_data = []
for i in range(TEST_DATASET_NUMBER):
    np.random.seed(i)
    test_data.append(np.array(df_test.sample(n=PADDING_SIZE-1, replace=True, random_state=i).append(df_test[df_test[0] == 2].sample(n=1, random_state=i)).reset_index(drop=True)))
    test_additional_data.append(np.asarray(calculate_classifications(test_data[i][:, 0], test_data[i][:, 1], test_data[i][:, np.r_[2:5]])))
    length = len(test_data[i])
    test_quantifications.append([np.count_nonzero(test_data[i][:, 0] == 0) / length, np.count_nonzero(test_data[i][:, 0] == 1) / length, np.count_nonzero(test_data[i][:, 0] == 2) / length])
    # data[i] = np.pad(data[i], ((0, PADDING_SIZE + MIN_SIZE - length), (0, 0)))
    test_data[i] = test_data[i][:, np.r_[2:2405]]
    test_data[i] = np.c_[test_data[i], - test_data[i][:, 0]*np.log(test_data[i][:, 0]) - test_data[i][:, 1]*np.log(test_data[i][:, 1]) - test_data[i][:, 2]*np.log(test_data[i][:, 2])]
    # Choose depending on what you want to sort: 0 - probability, 2403 - entropy
    test_data[i] = test_data[i][test_data[i][:, 2403].argsort()]
    # test_data[i] = test_data[i][test_data[i][:, 0].argsort()]
    print(i)

test_data = np.asarray(test_data)
test_additional_data = np.asarray(test_additional_data)
test_quantifications = np.asarray(test_quantifications)

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(PADDING_SIZE + MIN_SIZE, 2404), dtype="float32")
additional_inputs = keras.Input(shape=21, dtype="float32")
# Add a bidirectional LSTM
x = tf.keras.layers.Concatenate(axis=1)([layers.Bidirectional(layers.LSTM(128, return_sequences=False, input_shape=(PADDING_SIZE + MIN_SIZE, 2404)))(inputs), additional_inputs])
# Add dense layers
layer1 = layers.Dense(512, activation="relu", name="layer1")(x)
dropout1 = layers.Dropout(0.3)(layer1)
layer2 = layers.Dense(256, activation="relu", name="layer2")(x)
dropout2 = layers.Dropout(0.3)(layer2)
layer3 = layers.Dense(128, name="layer3")(dropout2)
dropout3 = layers.Dropout(0.3)(layer3)
layer4 = layers.Dense(64, name="layer4")(dropout3)
dropout4 = layers.Dropout(0.3)(layer4)
# Add a classifier
outputs = layers.Dense(3, activation="softmax")(dropout4)
model = keras.Model([inputs, additional_inputs], outputs)
model.summary()

model.compile("adam", "mean_absolute_error", metrics=["accuracy", "mean_absolute_error", "kullback_leibler_divergence"])
model.fit([data, additional_data], quantifications, batch_size=BATCH_SIZE, epochs=NUM_OF_EPOCHS, validation_data=([test_data, test_additional_data], test_quantifications))
model.save(f'service_padding{str(PADDING_SIZE)}_correct_2016_epoch{str(NUM_OF_EPOCHS)}_entropy_{str(DATASET_NUMBER)}_4layers_batch{str(BATCH_SIZE)}_last')
