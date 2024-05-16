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

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

plt.rcParams['font.family'] = 'NanumGothic'
# -

# # Series, DataFrame

# 시리즈 객체 값과 인덱스 확인
sr1 = pd.Series([10, 20, 30, 40, 50, 60])
print(sr1.index)
print(sr1.values)

# 시리즈는 리스트가 아닌 배열이기 때문에 인덱싱이 가능하지 않다. >> series의 값에 맞는 인덱스를 넣어줘야 한다.
# 음수 인덱싱을 하고 싶다면 리스트로 바꿔준 후 하면 가능하다. or iloc로 가능하다.
print(sr1[2])
list(sr1)[-2]

sr2 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(sr2)
print(sr2['b'])
print(sr2.iloc[2])
# 슬라이싱은 가능함..
print(sr2[1:4])

# DF 생성
pd.DataFrame([[10, 20, 30], [40, 50, 60]])

# 딕셔너리 활용 >> 데이터프레임 생성
dict1 = {
    'fruit' : ['apple', 'pear', 'orange', 'grape', 'banana'],
    'price' : [100, 120, 150, 50, 200],
    'qty' : [10, 25, 30, 20, 15]
}
df = pd.DataFrame(dict1, index=['a', 'b', 'c', 'd', 'e'])
df.rename(columns={'fruit' : '과일', 'price' : '가격', 'qty' : '수량'}, inplace=True)

df

df.rename(index={'a' : '대구', 'b' : '나주', 'c' : '제주', 'd' : '함양', 'e' : '아프리카'}, inplace=True)

df

# # 실데이터 분석

# +
path = '../datasets/df_sample.csv'

raw = pd.read_csv(path)
df = raw.copy()
# -

df.head(1)

# 요약 정보
print(df.shape)
df.info()

# 데이터 개수 세기 (컬럼별)
df.count()

# 기술통계
df.describe()

# 모든데이터에 대한 통계 보기
# (원래 describe는 수치형 데이터에 관한 것이지만, include='all'로 범주형 데이터도 확인 가능하다.)
df.describe(include='all')

df.iloc[:, 1:].mean()

df['학번'].min()

df[['중간', '기말']].agg(['min', 'max', 'median'])

# 각 컬럼의 표준편차와 분산
# std(), var()
df[['중간', '기말']].agg(['std', 'var'])

# 지정한 열이 포함되어 있는 고유한 값 개수 세기
# value_counts
df['퀴즈'].value_counts()

# 상관계수 구하기
df[['중간', '기말']].corr()

df.iloc[:, 1:].corr()

df.set_index('학번')

df.index = df['학번']

df.set_index('학번', inplace=True)

df.loc['S01', '중간']

df.iloc[1, 1]

# +
# df2 = df.drop(0)
# df2 = df2.drop('퀴즈', axis=1)
# df2
# -

df2 = df.sort_index(ascending=False)
df2

# DF 정렬 (특정 열 기준)
df = raw.copy()
df.sort_values(by='기말', ascending=False)

# ## 데이터 전처리

# 데이터 변환
df['합계'] = df['중간'] + df['기말']
df

x = ['1반', '1반', '1반', '2반', '2반', '2반', '3반', '3반', '3반', '3반']
df['class'] = x
df

df.insert(6,  'cls', x, True)

df

df['원래반'] = x
df.insert(6, '변경된 반', x, True)
df

# ## 표준화
# * 각 데이터 값이 평균을 가준으로 얼마나 떨어져 있나 (편차)
# * 평균(0), 편차 >> 표준편차(1)
# * z-score : (데이터값 - 평균) / 표준편차
# * 변환 후, 평균 0, 표준편차 1

# +
mid_avg = np.mean(df['중간'])
mid_std = np.std(df['중간'])

print(mid_avg, mid_std)
# -

df['mid_z_score'] = (df['중간'] - mid_avg) / mid_std

df.head(1)

# ## 정규화
# * 표준화는 정규화의 일종
# * 데이터의 범위를 0-1로 만드는 것 --> 확룰 때문에
# * (x - 최솟값) / (최댓값 - 최솟값) : 전체 범위 중 현재 값의 위치는?

# +
fin_min = np.min(df['기말'])
fin_max = np.max(df['기말'])

print(fin_min, fin_max)
# -

df['기말_정규화'] = (df['기말'] - fin_min) / (fin_max - fin_min)

df

df.iloc[:, -3:].groupby(['원래반']).mean()

# 각 범주별 빈도수 계산
df['중간'].groupby(df['cls']).count()

# 행 or 열에 지정된 함수 적용
# apply()
# 퀴즈에 루트 적용
df['퀴즈'].apply(np.sqrt)

# ## 결측값 처리

df.info()

out = [14, 15, 13, 14, None, None, 19, 11, 12, 18]
df['토론'] = out

df.info()

df.isnull().sum()

# * 결측값 대치
# 1. 단순 대치 (평균, 중앙값)
# 2. dropna()

df.dropna(axis=0)

dbt_mean = df['토론'].mean()

df['토론'].fillna(dbt_mean)

fin_q1 = df['기말'].quantile(0.25)
fin_q3 = df['기말'].quantile(0.75)
fin_iqr = fin_q3 - fin_q1
print(fin_iqr)

df.boxplot()

plt.figure(figsize=(10, 6))
plt.boxplot(df['기말'], labels=['기말'])
plt.title('기말고사')
plt.show()
