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

# # visualization

# +
path = '../data/raw_sales.csv'

pd.read_csv(path).head(3)
# -

raw = pd.read_csv(path)
df = raw.copy()
df.head(3)

# 불러올 때부터 datesold의 시간을 없애기
# parse_dates : 지정된 열을 날짜/시간 형식으로 자동 변환
# 변환된 'datesold' : datetime64 형식
raw = pd.read_csv(path, parse_dates=['datesold'])
df = raw.copy()
df.head(3)

df.shape

df.info() # 결측치가 없다는 것을 확인함.

# 연도만 추출
print(df['datesold'].dtype)
print(df['datesold'].dt.year)
df['year'] = df['datesold'].dt.year
df.head(3)

# 연도별 평균 가격
df.groupby('year')['price'].agg('mean')

# 정리
result = round(df.groupby('year')['price'].agg('mean'), 1)
result[:5]

print(result.index)
print(result.values)

# +
# 시간의 변동에 따른 추세 확인
# 선 그래프
# 그래프 바탕 설정
fig, ax = plt.subplots(figsize=(12, 6))

# 그래프 그리기
ax.plot(result.index, result.values)

# 제목, 축 이름 설정
ax.set_title('Average of Yearly Price', size=15)
ax.set_xlabel('year')
ax.set_ylabel('price')

# 그래프 저장하기
plt.savefig('../data/average_of_YP.png')

# 그래프만 보여주기
plt.show()
# -

# ## barplot

# 데이터 설정하기
df['month'] = df['datesold'].dt.month
df.head(3)

# +
# 해당 데이터가 있는지 확인하기
print(df['year'].isin([2007, 2008, 2009]))

# 조건에 넣기
condition = df['year'].isin([2007, 2008, 2009])
print(df.loc[condition, :].head(3))

# 변수 저장
df_condition = df.loc[condition, :]
df_condition.tail(3)

# +
# 연도별, 월별 평균 가격
df_condition.groupby(['year', 'month'])['price'].agg('mean') # 결과가 MultiIndex로 나온다.

# 평균을 소수점 첫째자리까지 표시하기
round(df_condition.groupby(['year', 'month'])['price'].agg('mean'), 1)

# 변수 설정
result = round(df_condition.groupby(['year', 'month'])['price'].agg('mean'), 1)
result[:5]
# -

# MultiIndex 를 분리해주기
result = result.reset_index()
result.head(3)

result['year'].unique()

# 2007년만 뽑기
result.loc[result['year'] == 2007, :]

# +
result_2007 = result.loc[result['year'] == 2007, :]
result_2008 = result.loc[result['year'] == 2008, :]
result_2009 = result.loc[result['year'] == 2009, :]

print(result_2007.head(3))
print(result_2008.head(3))
print(result_2009.head(3))

# +
# 그래프
# 바탕 만들기
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))

# 그래프 그리기
axes[0].bar(result_2007['month'], result_2007['price'])
axes[0].set_title('2007\'s Monthly Price Averages')

axes[1].bar(result_2008['month'], result_2008['price'])
axes[1].set_title('2008\'s Monthly Price Averages')

axes[2].bar(result_2009['month'], result_2009['price'])
axes[2].set_title('2009\'s Monthly Price Averages')

# 그래프 간 간격 맞춰주기
plt.tight_layout()

# 그래프 보여주기
plt.show()
# -








