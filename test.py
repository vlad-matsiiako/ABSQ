import numpy as np
import pandas as pd
import tensorflow as tf
from utils import calculate_classifications


DATASET_NUMBER = 100
PADDING_SIZE = 60
BATCH_SIZE = 32
NUM_OF_EPOCHS = 10
MIN_SIZE = 0


df_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_test_service', delimiter=" ", header=None)
# df_categories_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/data/programGeneratedData/categories_test_2016.csv', delimiter=",").drop(columns=['Unnamed: 0'])

df_train = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_train_service', delimiter=" ", header=None)
# df_categories_train = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/data/programGeneratedData/categories_train_2016.csv', delimiter=",").drop(columns=['Unnamed: 0'])

# calculate_classifications(np.asarray(df_train)[:, 0], np.asarray(df_train)[:, 1], np.asarray(df_train)[:, np.r_[2:5]])

df_test_all = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_test', delimiter=" ", header=None)
df_categories_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/data/programGeneratedData/categories_test_2016.csv', delimiter=",").drop(columns=['Unnamed: 0'])
df_restaurant_test = df_test_all[df_categories_test["Category"] == "SERVICE#GENERAL"].reset_index(drop=True)
df_test_all = df_restaurant_test
# calculate_classifications(np.asarray(df_test_all)[:, 0], np.asarray(df_test_all)[:, 1], np.asarray(df_test_all)[:, np.r_[2:5]])

data = []
quantifications = []
additional_data = []
for i in range(DATASET_NUMBER):
    np.random.seed(i)
    data.append(np.array(df_test.sample(n=PADDING_SIZE - 1, replace=True, random_state=i).append(df_test[df_test[0]==2].sample(n=1, random_state=i)).reset_index(drop=True)))
    additional_data.append(np.asarray(calculate_classifications(data[i][:, 0], data[i][:, 1], data[i][:, np.r_[2:5]])).reshape(1, 21))
    length = len(data[i])
    quantifications.append([np.count_nonzero(data[i][:, 0] == 0) / length, np.count_nonzero(data[i][:, 0] == 1) / length, np.count_nonzero(data[i][:, 0] == 2) / length])
    # data[i] = data[i][:, np.r_[2:2405]]
    data[i] = np.c_[data[i], - data[i][:, 2]*np.log(data[i][:, 2]) - data[i][:, 3]*np.log(data[i][:, 3]) - data[i][:, 4]*np.log(data[i][:, 4])]
    # Choose depending on what you want to sort: 0 - probability, 2405 - entropy
    data[i] = data[i][data[i][:, 2405].argsort()]
    print(i)

data_all = []
quantifications_all = []
additional_data_all = []
for i in range(DATASET_NUMBER):
    np.random.seed(i)
    data_all.append(np.array(df_test_all.sample(n=PADDING_SIZE - 1, replace=True, random_state=i).append(df_test_all[df_test_all[0]==2].sample(n=1, random_state=i)).reset_index(drop=True)))
    additional_data_all.append(np.asarray(calculate_classifications(data_all[i][:, 0], data_all[i][:, 1], data_all[i][:, np.r_[2:5]])).reshape(1, 21))
    length = len(data_all[i])
    quantifications_all.append([np.count_nonzero(data_all[i][:, 0] == 0) / length, np.count_nonzero(data_all[i][:, 0] == 1) / length, np.count_nonzero(data_all[i][:, 0] == 2) / length])
    # data[i] = data[i][:, np.r_[2:2405]]
    data_all[i] = np.c_[data_all[i], - data_all[i][:, 2]*np.log(data_all[i][:, 2]) - data_all[i][:, 3]*np.log(data_all[i][:, 3]) - data_all[i][:, 4]*np.log(data_all[i][:, 4])]
    # Choose depending on what you want to sort: 0 - probability, 2405 - entropy
    data_all[i] = data_all[i][data_all[i][:, 2405].argsort()]
    print(i)

data_all = np.asarray(data_all)
additional_data_all = np.asarray(additional_data_all)
quantifications_all = np.asarray(quantifications_all)


def calculate_losses(quantifications, real):
    ae = np.sum(np.abs(quantifications - real))/3
    rae = np.sum(np.abs(quantifications - real)/(real+0.0001))/3
    kld = np.sum(real * np.log((real+0.001)/(quantifications+0.0001)))

    return [ae, rae, kld]


def clip(values):
    if np.min(values) < 0:
        values = values - np.min(values)
        values = values / np.sum(values)
    return values


def clip2(values):
    if np.min(values) < 0:
        values = values / (np.sum(values) - np.sum(values[values < 0]))
    values[values < 0] = 0
    return values


results_cc = pd.DataFrame(columns=['AE', 'RAE', 'KLD'])
results_pcc = pd.DataFrame(columns=['AE', 'RAE', 'KLD'])
results_acc = pd.DataFrame(columns=['AE', 'RAE', 'KLD'])
results_pacc = pd.DataFrame(columns=['AE', 'RAE', 'KLD'])
results_quanet = pd.DataFrame(columns=['AE', 'RAE', 'KLD'])
results_quanet_entropy = pd.DataFrame(columns=['AE', 'RAE', 'KLD'])
results_aspquanet = pd.DataFrame(columns=['AE', 'RAE', 'KLD'])
model = tf.keras.models.load_model('service_padding60_correct_2016_epoch3_entropy_4000_4layers_batch32_last')
# food_padding150_correct_2016_epoch10_entropy_5000_4layers_batch32_new
# service_padding60_correct_2016_epoch3_entropy_4000_4layers_batch32_last
# ambience_padding30_correct_2016_epoch4_entropy_4000_4layers_batch32_1
# restaurant_padding30_correct_2016_epoch3_entropy_2200_4layers_batch16_new
for i in range(DATASET_NUMBER):
    results = calculate_classifications(data[i][:, 0], data[i][:, 1], data[i][:, np.r_[2:5]])[9:]
    results_cc = results_cc.append(pd.Series(calculate_losses(np.asarray(results[0:3]), np.asarray(quantifications[i])), index = results_cc.columns), ignore_index=True)
    results_pcc = results_pcc.append(pd.Series(calculate_losses(np.asarray(results[3:6]), np.asarray(quantifications[i])), index = results_pcc.columns), ignore_index=True)
    results_acc = results_acc.append(pd.Series(calculate_losses(clip(np.asarray(results[6:9])), np.asarray(quantifications[i])), index = results_acc.columns), ignore_index=True)
    results_pacc = results_pacc.append(pd.Series(calculate_losses(clip(np.asarray(results[9:12])), np.asarray(quantifications[i])), index = results_pacc.columns), ignore_index=True)
    a = model.predict([np.expand_dims(data[i][:, np.r_[2:2406]], 0), additional_data[i]])[0]
    b = calculate_losses(a, np.asarray(quantifications[i]))
    results_aspquanet = results_aspquanet.append(pd.Series(b, index=results_aspquanet.columns), ignore_index=True)

model = tf.keras.models.load_model('allservice_padding60_correct_2016_epoch5_entropy_4000_4layers_batch16_3')
# allfood_padding150_correct_2016_epoch10_entropy_5000_4layers_batch32_1
# allservice_padding60_correct_2016_epoch5_entropy_4000_4layers_batch16_3
# allambience_padding30_correct_2016_epoch4_entropy_5000_4layers_batch32_1
# allrestaurant_padding30_correct_2016_epoch3_entropy_4000_4layers_batch32_1
for i in range(DATASET_NUMBER):
    results_all = calculate_classifications(data_all[i][:, 0], data_all[i][:, 1], data_all[i][:, np.r_[2:5]])[9:]
    results_quanet_entropy = results_quanet_entropy.append(pd.Series(calculate_losses(model.predict([np.expand_dims(data_all[i][:, np.r_[2:2406]], 0), additional_data_all[i]])[0], np.asarray(quantifications_all[i])), index=results_quanet.columns), ignore_index=True)

for i in range(DATASET_NUMBER):
    data_all[i] = data_all[i][data_all[i][:, 1].argsort()]
    print(i)

model = tf.keras.models.load_model('allservice_padding60_correct_2016_epoch5_firstprob_5000_4layers_batch32_1')
# allfood_padding150_correct_2016_epoch1_firstprob_5000_3layers_batch32_2
# allservice_padding60_correct_2016_epoch5_firstprob_5000_4layers_batch32_1
# allambience_padding30_correct_2016_epoch5_firstprob_5000_4layers_batch32_1
# allrestaurant_padding30_correct_2016_epoch1_firstprob_4000_4layers_batch32_1
for i in range(DATASET_NUMBER):
    results_all = calculate_classifications(data_all[i][:, 0], data_all[i][:, 1], data_all[i][:, np.r_[2:5]])[9:]
    results_quanet = results_quanet.append(pd.Series(calculate_losses(model.predict([np.expand_dims(data_all[i][:, np.r_[2:2406]], 0), additional_data_all[i]])[0], np.asarray(quantifications_all[i])), index=results_quanet.columns), ignore_index=True)


results_cc = results_cc.mean()
results_acc = results_acc.mean()
results_pcc = results_pcc.mean()
results_pacc = results_pacc.mean()
results_quanet = results_quanet.mean()
results_quanet_entropy = results_quanet_entropy.mean()
results_aspquanet = results_aspquanet.mean()
results_losses = pd.DataFrame()
results_losses["CC"] = results_cc
results_losses["ACC"] = results_acc
results_losses["PCC"] = results_pcc
results_losses["PACC"] = results_pacc
results_losses["QuaNet"] = results_quanet
results_losses["QuaNet Entropy"] = results_quanet_entropy
results_losses["AspQuaNet"] = results_aspquanet


model = tf.keras.models.load_model('all_data_2016_epoch10_food_adam')
