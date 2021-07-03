import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

SIZE = (8, 2.5)


df = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/training.txt', ", ")
df_test = pd.read_csv('/Users/vmatsiiako/Desktop/Erasmus/Thesis/Code/prob1.txt_outputs_fin_test_service', delimiter=" ", header=None)

plt.plot(df["epoch"], df["kld"], color='blue', marker='o', label="Training Set")
plt.plot(df["epoch"], df["val_kld"], color='green', marker='o', label="Validation Set")
plt.title('KLD', fontsize=14)
plt.xlabel('Epoch #', fontsize=14)
plt.ylabel('Kullback-Leibler divergence', fontsize=14)
plt.grid(True)
plt.legend(loc='best')
plt.show()

df_food_test_neutral = df_test[df_test[0] == 2].sample(n=50, replace=True, random_state=1)
df_food_test_negative = df_test[df_test[0] == 1].sample(n=50, replace=True, random_state=1)
df_food_test_positive = df_test[df_test[0] == 0].sample(n=50, random_state=1)

df_test = pd.concat([df_food_test_positive, df_food_test_negative, df_food_test_neutral])


df_test[2405] = df_test[2] * np.log(df_test[2]) + df_test[3] * np.log(df_test[3]) + df_test[4] * np.log(df_test[4])
df_test = df_test.sort_values(by=[2405]).reset_index(drop=True)

data = pd.DataFrame()
data['index'] = df_test.index
data['0'] = df_test[2]
data['1'] = df_test[3]
data['2'] = df_test[4]

plt.figure(figsize=SIZE)
plot1 = plt.figure(1)
p1 = plt.bar(data['index'], data['0'], 1, color='b', label='Positive')
p2 = plt.bar(data['index'], data['1'], 1, bottom=data['0'], color='g', label='Negative')
p3 = plt.bar(data['index'], data['2'], 1, bottom=np.array(data['0'])+np.array(data['1']), color='r', label='Neutral')
plt.legend()

plt.figure(figsize=SIZE)
plot2 = plt.figure(2)
ones = np.ones(len(df_test[0]))
values = np.array(df_test[0])
idx = np.array(df_test.index)
clrs = ['b' if (x == 0) else 'g' if (x == 1) else 'r' for x in values]
plt.bar(idx, ones, color=clrs, width=1)
plt.show()