import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

path = os.getcwd()

df = pd.read_excel(path + "/data/risk.xlsx")

label = np.array(df['高/低危组'])

high_index = np.argwhere(label == 1).flatten()
low_index = np.argwhere(label == 0).flatten()
# print(high_index)
# print(low_index)

siri = np.array(df['前SIRI']).reshape(-1, 1)
lmr = np.array(df["前LMR"]).reshape(-1, 1)
plr = np.array(df['前PLR']).reshape(-1, 1)
nlr = np.array(df["前NLR"]).reshape(-1, 1)

print('前SIRI高低危组平均值：', sum(siri[high_index])/len(high_index), sum(siri[low_index])/len(low_index))
print('前LMR高低危组平均值：', sum(lmr[high_index])/len(high_index), sum(lmr[low_index])/len(low_index))
print('前PLR高低危组平均值：', sum(plr[high_index])/len(high_index), sum(plr[low_index])/len(low_index))
print('前NLR高低危组平均值：', sum(nlr[high_index])/len(high_index), sum(nlr[low_index])/len(low_index))

plt.plot(siri[high_index], color='r', linewidth=2, linestyle='dashed', label='前SIRI高危组')
plt.plot(siri[low_index], color='g', linewidth=2, linestyle='dashed', label='前SIRI低危组')
plt.legend(loc='upper right')
plt.show()

plt.plot(lmr[high_index], color='r', linewidth=2, linestyle='dashed', label='前LMR高危组')
plt.plot(lmr[low_index], color='g', linewidth=2, linestyle='dashed', label='前LMR低危组')
plt.legend(loc='upper right')
plt.show()

plt.plot(plr[high_index], color='r', linewidth=2, linestyle='dashed', label='前PLR高危组')
plt.plot(plr[low_index], color='g', linewidth=2, linestyle='dashed', label='前PLR低危组')
plt.legend(loc='upper right')
plt.show()

plt.plot(nlr[high_index], color='r', linewidth=2, linestyle='dashed', label='前NLR高危组')
plt.plot(nlr[low_index], color='g', linewidth=2, linestyle='dashed', label='前NLR低危组')
plt.legend(loc='upper right')
plt.show()