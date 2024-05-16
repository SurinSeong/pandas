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

# # Series & DataFrame

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# +
s = pd.Series([1, 3, 6, 9])

print(s) # index 달고 나옴
print(s.index)
print(s.values) # 리스트 형식으로 출력
# -

# 순서를 만들려면 list로 변경해야 한다.
list(s.values)

dict(s) # index : value 형태

for k, v in s.items():
    print(f'{k} : {v}')

[k for k, v in s.items()]

# +
# 리스트를 DF로 만들기
v = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
i = ['첫 번째 행', '두 번째 행', '세 번째 행']
c = ['컬럼1', '컬럼2', '컬럼3']

pd.DataFrame(v, index=i, columns=c)

# +
data = [
    ['100', '강백호', 9.7],
    ['101', '서태웅', 8.9],
    ['102', '송태섭', 9.3],
    ['103', '서장훈', 7.1]
]

pd.DataFrame(data)
# -

# ## 예제

# +
# 파일 불러오기
file_path = '../datasets/california_housing_train.csv'

raw = pd.read_csv(file_path)
df = raw.copy()
# -

print(df.shape)
df.columns

# +
# 위, 경도는 folium으로 좌표를 찍을 수 있음.
# -

df.info()

df.head(1)

df.describe()

housing_median_age = df['housing_median_age']

housing_median_age.value_counts()

# 데이터프레임 컬럼 다루기
# 생성 or 수정
df.columns

# 새로운 컬럼 만들기
df['age_0'] = 0

df['age_10'] = housing_median_age // 10

df.head(1)

df.columns[-2:]

# df = df.drop('age_0', axis=1)
# df = df.drop('age_10', axis=1)
df.drop(df.columns[-2:], axis=1, inplace=True)

df.head(1)

# DF 데이터 조회
housing_median_age.head(2)

housing_median_age[housing_median_age == 30] # condition 중요 !!

condition = ((housing_median_age > 30) & (df['total_rooms'] < 100) & (df['median_income'] > 10)) # condition 하나로 하기 싫으면 쪼개도 됨.
df[condition]

# ### loc, iloc

df.iloc[0, 2]

df.loc[0, 'housing_median_age']

df.loc[housing_median_age == 30, ['housing_median_age', 'total_bedrooms']]

# # 데이터 프레임 정렬과 집계

# housing_median_age를 기준으로 정렬하기
housing_sorted = df.sort_values(by=['housing_median_age']) # default : 오름차순

housing_sorted.head(1)

# 컬럼 순으로 정렬을 해준다.
df.sort_values(by=['housing_median_age', 'total_rooms'], ascending=[False, True])

# ~ 별
df.groupby(['housing_median_age'])[['total_rooms', 'total_bedrooms']].mean()

df.groupby(['housing_median_age'])['total_rooms'].agg(['mean', 'max', 'sum'])

# DF 결측치 처리
df['age_na'] = np.nan

df.isnull().sum()

df['age_na'] = df['age_na'].fillna(round(df['housing_median_age'].mean(), 2))

df.head(1)




























