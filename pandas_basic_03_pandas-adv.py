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
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.DataFrame(np.random.rand(5, 2), columns=['A', 'B'])
print(df['A'])
print(df['A'] < 0.5) # bool 값으로 나온다.
print(df[(df['A'] < 0.5) & (df['B'] > 0.3)])

# +
# 조건을 변수에 넣어 조금 더 확인하기 쉽게 한다.
condition = (df['A'] < 0.5) & (df['B'] > 0.3)
print(df[condition])

# df.query() 사용
print(df.query('(A < 0.5) and (B > 0.3)'))

# +
# 문자열 조건 검색
data = {
    'Animal' : ['Dog', 'Cat', 'Cat', 'Pig', 'Cat'],
    'Name' : ['Happy', 'Sam', 'Tom', 'Mini', 'Rocky']
}

# 데이터프래임 만들기
df = pd.DataFrame(data)
df.head(3)
# -

df['Animal']

# 'Animal' 컬럼 중 'Cat'이 포함되어있는지 bool 문자로 반환
df['Animal'].str.contains('Cat')

df[df['Animal'].str.contains('Cat')]

# Q. 문자열 검색 시 대소문자 상관없이 결과값 나오게 하기 --> case=False 사용
df

df['Animal'].str.contains('cat', case=False)

# +
# 함수로 데이터 처리하기
df = pd.DataFrame(np.arange(5), columns=['num'])

# 제곱하는 함수 만들기
def square(x):
    return x ** 2

print(df['num'])

# apply를 적용하는 것은 f(g(x))랑 같은 의미임.
df['num'].apply(square)
# -

df['square'] = df['num'].apply(square)
print(df)

df.num # df['num']과 동일

df['square_lambda'] = df.num.apply(lambda x : x ** 2)
df

# +
# 함수로 데이터 처리하기
df = pd.DataFrame(columns=['phone'])

# 데이터 삽입하기 (loc 사용)
df.loc[0] = '010-1234-5678'
df.loc[1] = '공일공-일이삼사-오육칠팔'
df.loc[2] = '010.1234.오육칠팔'
df.loc[3] = '공1공-1234-5육7팔'

# 새로운 컬럼 만들기 (공란)
df['preprocess_phone'] = ''
df


# -

# 전처리용 함수 만들기
def preprocess(phone):
    mapping = {
        '공' : '0',
        '일' : '1',
        '이' : '2',
        '삼' : '3',
        '사' : '4',
        '오' : '5',
        '육' : '6',
        '칠' : '7',
        '팔' : '8',
        '.' : '',
        '-' : ''
    }
    for k, v in mapping.items():
        phone = phone.replace(k, v)
    return phone


df['phone']

df.preprocess_phone = df['phone'].apply(preprocess)
df['preprocess_phone']

df

# +
# replace : apply 기능에서 데이터 값만 대체하고 싶을 때 사용함
data = {
    '0' : 'Male',
    '1' : 'Male',
    '2' : 'Female',
    '3' : 'Female',
    '4' : 'Male'
}

# 데이터프레임 만들기
data_list = list(data.values())
df = pd.DataFrame(data_list, columns=['Gender'])
df
# -

# label encoding (0, 1)
df.Gender.replace({'Male' : 0, 'Female' : 1}, inplace=True) # 업데이트를 해줘야 원래의 df에 저장이 된다.

df

# 그룹으로 묶기
# 조건부로 집계하고 싶은 경우
df = \
pd.DataFrame({'key': ['a', 'b', 'c', 'a', 'b', 'c'],
              'data' : range(6)})
df.head(3)

# key를 기준으로 묶어서 값 보여주기
df.groupby('key').sum()

df.groupby(['key', 'data']).sum()

# aggregate : groupby를 통해 집계를 한번에 계산
data = {
    'group' : ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'x' : [20, 30, 40, 20, 30, 40, 50, 30],
    'y' : [1, 2, 3, 4, 5, 6, 7, 8]
}
df = pd.DataFrame(data)
df.head(3)

df.groupby('group').agg(['min', 'median', 'max'])

df.groupby('group').agg({'x' : 'min', 'y' : 'sum'})


# * 4/26

# +
# filter : groupby를 통해 그룹으로 묶은 상태에서 그룹 속성을 기준으로 데이터를 필터링
def filter_mean(x):
    return x['y'].mean() > 3
    
print(df.groupby('group').mean())
print(df.groupby('group').filter(filter_mean)) # 'group'을 기준으로 나눈 'y'들의 평균이 3보다 크기 때문에 모든 데이터가 추출되었음.

# +
# apply : groupby를 통해 묶인 데이터에 함수 적용

df.groupby('group').apply(lambda x : x.max()-x.min()) # 최대  - 최소 : 중요 !!!
# -

# get_group : groupby로 묶인 데이터에서 key값으로 데이터 가져오기
df.groupby('group').get_group('A')

# MultiIndex : index 자체에 계층을 만든다.
df = \
pd.DataFrame(
    np.random.randn(4, 2),
    index=[['A', 'A', 'B', 'B'], [1, 2, 1, 2]],
    columns=['x', 'y']
)
df

df['x']

df_col = \
pd.DataFrame(
    np.random.randn(4, 4),
    columns=[['A', 'A', 'B', 'B'], ['x', 'y', 'x', 'y']]
)
df_col

df_col['A']

df_col['A']['x']

data = {
    "날짜": ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-02"],
    "카테고리": ["과일", "채소", "채소", "과일", "과일", "채소"],
    "수량": [10, 15, 5, 20, 10, 5],
    '판매처' : ['서울', '부산', '제주', '서울', '부산', '제주']
}
df = pd.DataFrame(data)
df

# pivot_table : index(행 index로 들어갈 key) / columns(열 index로 라벨링되는 값) / values(분석할 데이터 값)
df.pivot_table(
    values='수량',
    index='날짜',
    columns='카테고리',
    aggfunc='sum'
)

df.pivot_table(
    values='수량',
    index='날짜',
    columns='카테고리',
    aggfunc='mean'
)

df.pivot_table(
    values='수량',
    index='날짜',
    columns=['카테고리', '판매처'],
    aggfunc='median',
    fill_value=0
)


