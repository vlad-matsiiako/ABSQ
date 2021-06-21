import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from utils import calculate_classifications


# mnist = keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
DATASET_NUMBER = 10000
PADDING_SIZE = 300
BATCH_SIZE = 32
NUM_OF_EPOCHS = 10
MIN_SIZE = 0


df_train = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_train', delimiter=" ", header=None)
df_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_test_first', delimiter=" ", header=None)
df_categories_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/data/programGeneratedData/categories_test_2016.csv', delimiter=",").drop(columns=['Unnamed: 0'])
df_categories_train = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/data/programGeneratedData/categories_train_2016.csv', delimiter=",").drop(columns=['Unnamed: 0'])



# df_categories_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/data/programGeneratedData/BERT768testdata2016.txt', delimiter="\n", header=None)
df_food_test = df_test[df_categories_test["Category"] == "FOOD#QUALITY"].reset_index(drop=True)
df_service_test = df_test[df_categories_test["Category"] == "SERVICE#GENERAL"].reset_index(drop=True)
df_ambience_test = df_test[df_categories_test["Category"] == "AMBIENCE#GENERAL"].reset_index(drop=True)
df_restaurant_test = df_test[df_categories_test["Category"] == "RESTAURANT#GENERAL"].reset_index(drop=True)

df_food_train = df_train[df_categories_train["Category"] == "FOOD#QUALITY"].reset_index(drop=True)
df_service_train = df_train[df_categories_train["Category"] == "SERVICE#GENERAL"].reset_index(drop=True)
df_ambience_train = df_train[df_categories_train["Category"] == "AMBIENCE#GENERAL"].reset_index(drop=True)
df_restaurant_train = df_train[df_categories_train["Category"] == "RESTAURANT#GENERAL"].reset_index(drop=True)

# df_train = df_food_train

data = []
quantifications = []
additional_data = []
for i in range(DATASET_NUMBER):
    np.random.seed(i)
    # np.round(np.random.rand(1), 2)[0]
    # frac=len(df_test[0])/len(df_train[0])
    data.append(np.array(df_train.sample(n=int(np.random.rand()*PADDING_SIZE) + 1 + MIN_SIZE, replace=False, random_state=i).reset_index(drop=True)))
    additional_data.append(calculate_classifications(data[i][:, 0], data[i][:, 1], data[i][:, np.r_[2:5]]))
    length = len(data[i])
    quantifications.append([np.count_nonzero(data[i][:, 0] == 0) / length, np.count_nonzero(data[i][:, 0] == 1) / length, np.count_nonzero(data[i][:, 0] == 2) / length])
    data[i] = np.pad(data[i], ((0, PADDING_SIZE + MIN_SIZE - length), (0, 0)))
    data[i] = data[i][:, np.r_[2:2405]]
    data[i] = data[i][data[i][:, 0].argsort()]
    print(i)


data = np.asarray(data)
additional_data = np.asarray(additional_data)
quantifications = np.asarray(quantifications)

df_test = df_food_test
x_test_food = np.asarray(df_test.iloc[:, np.r_[2:2405]])
y_test_food = np.asarray([len(df_test[df_test[0] == 0])/len(df_test[2]), len(df_test[df_test[0] == 1])/len(df_test[3]), len(df_test[df_test[0] == 2])/len(df_test[4])]).reshape(1, 3)
x_test_additional_food = np.asarray(calculate_classifications(df_test.iloc[:, 0], df_test.iloc[:, 1], df_test.iloc[:, np.r_[2:5]])).reshape(1, 21)
x_test_food = np.pad(x_test_food, ((0, PADDING_SIZE + MIN_SIZE - len(df_test)), (0, 0)))
x_test_food = np.expand_dims((x_test_food[x_test_food[:, 0].argsort()]), 0)

# df_test = df_service_test
# x_test_service = np.asarray(df_test.iloc[:, np.r_[2:2405]])
# y_test_service = np.asarray([len(df_test[df_test[0] == 0])/len(df_test[2]), len(df_test[df_test[0] == 1])/len(df_test[3]), len(df_test[df_test[0] == 2])/len(df_test[4])]).reshape(1, 3)
# x_test_additional_service = np.asarray(calculate_classifications(df_test.iloc[:, 0], df_test.iloc[:, 1], df_test.iloc[:, np.r_[2:5]])).reshape(1, 21)
# x_test_service = np.pad(x_test_service, ((0, PADDING_SIZE + MIN_SIZE - len(df_test)), (0, 0)))
# x_test_service = np.expand_dims((x_test_service[x_test_service[:, 0].argsort()]), 0)

# df_test = df_ambience_test
# x_test_ambience = np.asarray(df_test.iloc[:, np.r_[2:2405]])
# y_test_ambience = np.asarray([len(df_test[df_test[0] == 0])/len(df_test[2]), len(df_test[df_test[0] == 1])/len(df_test[3]), len(df_test[df_test[0] == 2])/len(df_test[4])]).reshape(1, 3)
# x_test_additional_ambience = np.asarray(calculate_classifications(df_test.iloc[:, 0], df_test.iloc[:, 1], df_test.iloc[:, np.r_[2:5]])).reshape(1, 21)
# x_test_ambience = np.pad(x_test_ambience, ((0, PADDING_SIZE + MIN_SIZE - len(df_test)), (0, 0)))
# x_test_ambience = np.expand_dims((x_test_ambience[x_test_ambience[:, 0].argsort()]), 0)

# df_test = df_restaurant_test
# x_test_restaurant = np.asarray(df_test.iloc[:, np.r_[2:2405]])
# y_test_restaurant = np.asarray([len(df_test[df_test[0] == 0])/len(df_test[2]), len(df_test[df_test[0] == 1])/len(df_test[3]), len(df_test[df_test[0] == 2])/len(df_test[4])]).reshape(1, 3)
# x_test_additional_restaurant = np.asarray(calculate_classifications(df_test.iloc[:, 0], df_test.iloc[:, 1], df_test.iloc[:, np.r_[2:5]])).reshape(1, 21)
# x_test_restaurant = np.pad(x_test_restaurant, ((0, PADDING_SIZE + MIN_SIZE - len(df_test)), (0, 0)))
# x_test_restaurant = np.expand_dims((x_test_restaurant[x_test_restaurant[:, 0].argsort()]), 0)

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(PADDING_SIZE + MIN_SIZE, 2403), dtype="float32")  # Embed each integer in a 128-dimensional vector
additional_inputs = keras.Input(shape=21, dtype="float32")
# x = layers.Embedding(195, 2401)(inputs)
# x = layers.Flatten()(inputs)
# Add a bidirectional LSTM
# x = tf.concat([layers.Bidirectional(layers.LSTM(128, return_sequences=False, input_shape=(PADDING_SIZE + MIN_SIZE, 2403)))(inputs), additional_inputs], 1)
x = tf.keras.layers.Concatenate(axis=1)([layers.Bidirectional(layers.LSTM(128, return_sequences=False, input_shape=(PADDING_SIZE + MIN_SIZE, 2403)))(inputs), additional_inputs])
# x = layers.Flatten()(x)
# Add dense layers
layer1 = layers.Dense(256, activation="relu", name="layer1")(x)
dropout1 = layers.Dropout(0.3)(layer1)
layer2 = layers.Dense(128, activation="relu", name="layer2")(dropout1)
dropout2 = layers.Dropout(0.3)(layer2)
layer3 = layers.Dense(64, name="layer3")(dropout2)
dropout3 = layers.Dropout(0.3)(layer3)
# Add a classifier
outputs = layers.Dense(3, activation="softmax")(dropout3)
model = keras.Model([inputs, additional_inputs], outputs)
model.summary()

model.compile("adam", "mean_absolute_error", metrics=["accuracy", "mean_absolute_error", "kullback_leibler_divergence"])
model.fit([data, additional_data], quantifications, batch_size=BATCH_SIZE, epochs=NUM_OF_EPOCHS, validation_data=([x_test_food, x_test_additional_food], y_test_food))
model.save('all_data_2016_epoch10_overnight')
# tf.saved_model.save(model, 'all_data_2016_epoch10')
# model.evaluate([x_test_service, x_test_additional_service], y_test_service)
# model.evaluate([x_test_ambience, x_test_additional_ambience], y_test_ambience)
# model.evaluate([x_test_restaurant, x_test_additional_restaurant], y_test_restaurant)
# tf.keras.metrics.MeanRelativeError(),


