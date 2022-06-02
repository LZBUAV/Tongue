from asyncio.windows_events import NULL
import pandas as pd
import os
import numpy as np
from sqlalchemy import column

path = os.getcwd()

df1 = pd.read_excel(path + '/data/1.xlsx')
df2 = pd.read_csv(path + '/data/2.csv')

df1_col0_name = df1.columns.to_list()
df1_col0 = df1[df1_col0_name[0]].to_list()

df2_row_name = df2.columns.to_list()
df2_row_last = np.array(df2.iloc[0])

new_clo = df1_col0_name.copy()
new_clo.append('X')

# print(df1_col0_name)
# print(new_clo)

df1.reindex(columns = new_clo)
df1['X'] = NULL
# print(df1.head())

df1_dict = []
for i in df1.index.values:
    row_data = df1.iloc[i]
    df1_dict.append(row_data)

# print(len(df2_row_name))
# print(df2_row_last.shape)

num = 381
for i in range(1, len(df2_row_name)):
    if df2_row_name[i] in df1_col0:
        i_index = df1_col0.index(df2_row_name[i])
        df1_dict[i_index]['X'] = df2_row_last[i]
    else:
        # i_dict = df1_dict[-1].copy()
        # i_dict['X'] = df2_row_last[i]
        # i_dict = {381:{'X': df2_row_last[i]}}
        # df1_dict.append(i_dict)
        df1.append({'X':df2_row_last[i]}, ignore_index=True)

# print(df1_dict[-1])
# print(len(df1_dict))

# df3 = pd.DataFrame(df1_dict)
df1.to_excel(path + '/data/3.xlsx')