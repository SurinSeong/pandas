# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd

# ## Selection by label

# +
# at()

df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
                  index = [4, 5, 6], columns = ['A', 'B', 'C'])
df
# -

df.at[4, 'B']

df.at[4, 'B'] = 10
df.at[4, 'B']

df.loc[5].at['B']

# ## Selection by position
# * iloc[행, 열]
# * iat[]

mydict = [{'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4},
          {'a' : 100, 'b' : 200, 'c' : 300, 'd' : 400},
          {'a' : 1000, 'b' : 2000, 'c' : 3000, 'd' : 4000}]
df = pd.DataFrame(mydict)
df

type(df.iloc[0])

df.iloc[0]

df.iloc[[0]]

type(df.iloc[[0]])

df.iloc[[0, 1]]

df.iloc[:3]

df.iloc[[True, False, True]]

df.iloc[lambda x : x.index % 2 == 0]

df.iloc[0, 1]

df.iloc[[0, 2], [1, 3]]

df.iloc[1:3, 0:3]

df.iloc[:, [True, False, True, False]]

df.iloc[:, lambda df : [0, 2]]

df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
                  columns=['A', 'B', 'C'])
df

df.iat[1, 2]

df.iat[1, 2] = 10
df.iat[1, 2]

df.loc[0].iat[1]

# * isin([])

# +
s = pd.Series(['llama', 'cow', 'llama', 'beetle', 'llama', 'hippo'],
              name = 'animal')

# [] 안의 내용을 가지고 있으면 True, 아니면 False
s.isin(['cow', 'llama'])
# -

# ~ : 반대
~s.isin(['cow', 'llama'])

s.isin(['llama'])

# 숫자와 문자는 같지 않음.
pd.Series([1]).isin(['1'])

pd.Series([1.1]).isin(['1.1'])

# # Missing Data

# * dropna()

df = pd.DataFrame({'name' : ['Alfred', 'Batman', 'Catwoman'],
                   'toy' : [np.nan, 'Batmobile', 'Bullwhip'],
                   'born' : [pd.NaT, pd.Timestamp('1940-04-25'),
                             pd.NaT]})
df

df.dropna()

df.dropna(axis='columns')

df.dropna(how='all')

df.dropna(thresh = 2)

df.dropna(subset=['name', 'toy'])

# * fillna()

df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                  [3, 4, np.nan, 1],
                  [np.nan, np.nan, np.nan, np.nan],
                  [np.nan, 3, np.nan, 4]],
                  columns=list('ABCD'))
df

# NaN 값을 () 안의 값으로 대체
df.fillna(0)

values = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3}
df.fillna(value = values)

# 하나의 NaN에만 값 채우기
df.fillna(value=values, limit=1)

df2 = pd.DataFrame(np.zeros((4, 4)), columns=list('ABCD'))
df.fillna(df2)


