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

# # 서울시 물가정보 분석

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# 한글 폰트 오류 해결하기
from matplotlib import font_manager, rc
import matplotlib as mpl

plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 깨짐 현상 해결
mpl.rcParams['axes.unicode_minus'] = False

# +
# 데이터 수집
path = '../datasets/생필품 농수축산물 가격 정보(2021년1월_6월).csv'

raw = pd.read_csv(path, encoding='cp949')
df = raw.copy()
# -

# 결측치 확인
df.isnull().sum()

# +
# 데이터 타입 확인
df.dtypes

# 년도-월 : 시계열 데이터가 필요하다면 날짜 데이터로 바꿔주기
# 점검일자 : 상동
# -

# 컬럼명 확인
df.columns

# 데이터 확인
# 컬럼별 속성정보 확인
df['시장/마트 번호'].unique() # 102개의 종류의 데이터 존재

# 시장/마트 이름 확인
df['시장/마트 이름'].unique() # 번호와 개수 같음. 누락된 것 없음.

# * 겹치는 것이 있다면 drop_duplicates() 사용하면 된다.

# 자치구별 시장/마트 개수 확인용 DF 만들기
df_market = df[['시장/마트 번호', '시장/마트 이름', '자치구 이름', '시장유형 구분(시장/마트) 이름']].drop_duplicates()

df_market.head(2)

df_market['자치구 이름'].value_counts()

# +
# 조건 설정
condition = df_market['자치구 이름'] == '관악구'

df_market[condition]

# +
# 조건 설정
condition = df_market['자치구 이름'] == '구로구'

df_market[condition]
# -

# 품목 번호
df.head(1)

df_items = df[['품목 번호', '품목 이름']].drop_duplicates()

df_items = df_items.sort_values('품목 이름')

# 품목 이름 확인 (30개씩)
df_items[:30]

df_items[30:60]

df_items[60:]

# 자치구 목록
df_gu = df[['자치구 코드', '자치구 이름']].drop_duplicates()

df_gu.shape

# 시장 유형
df_gubun = df[['시장유형 구분(시장/마트) 코드', '시장유형 구분(시장/마트) 이름']].drop_duplicates()

# ## 삼겹살 가격 분석

# +
# 조건 설정
condition = (df['품목 이름'].str.contains('삼겹살')) & (df['년도-월'] == '2021-06')

# df에서 조건에 맞는 데이터 불러오기
df[condition]

# +
# 조건 설정
condition = (df['품목 이름'].str.contains('삼겹살')) & (df['년도-월'] == '2021-06') & (df['실판매규격'].str.contains('600g'))

# df에서 조건에 맞는 데이터 불러오기
df_sam = df[condition]
# -

# 삼겹살 600g의 평균 가격은?
df_sam['가격(원)'].agg(['mean', 'max', 'min', 'median'])

# 가장 비싼 곳은 어디?
df_sam[df_sam['가격(원)'] == 35890]

# 가장 싼 곳은?
df_sam[df_sam['가격(원)'] == 1690]

# 내가 원하는 조건의 삼겹살이 있을까?
df_sam[df_sam['가격(원)'] < 5000]

# +
# 우리 동네 삼겹살 가격?
gu = input('내가 사는 동네 : ')

condition = (df_sam['자치구 이름'] == gu)
# -

df_sam_gu = df_sam[condition][['시장/마트 이름', '품목 이름', '실판매규격', '가격(원)']]

# +
# 시각화
x = df_sam_gu['시장/마트 이름']
y = df_sam_gu['가격(원)']

plt.scatter(x, y)
plt.title('광진구 삼겹살(600g) 가격')
plt.grid(True)
plt.show()
# -

# 마트 지점별 삼겹살 가격
# 유형 선택
mart = input('시장/마트 이름 : ')
condition = df_sam['시장/마트 이름'].str.contains(mart)
df_sam_mart = df_sam[condition]

df_sam_mart[['시장/마트 이름', '품목 이름', '실판매규격', '가격(원)']]

# +
# 시각화

x = df_sam_mart['시장/마트 이름']
y = df_sam_mart['가격(원)']

plt.scatter(x, y)
plt.title('이마트 지점별 삼겹살(600g) 가격')
plt.grid(True)
plt.xticks(rotation=60)
plt.show()

# -

# # In-class Practice
# * 미션 : 달걀 가격 분석
# * 조건 : 달걀, 30개
# * 어떤 구?

df.info()

df.head(1)

condition = (df['품목 이름'].str.contains('달걀')) & (df['실판매규격'].str.contains('30개')) & (df['년도-월'] == '2021-06')
df_egg30 = df[condition]

df_egg30['실판매규격'].unique()

# 평균, 최대, 최소 구하기
df_egg30['가격(원)'].agg(['mean', 'max', 'min'])

# +
# 이상치 시각화로 확인해보기
x = df_egg30['가격(원)']

plt.boxplot(x)
plt.xlabel('달걀(30개)')
plt.show()
# -

# 최소 가격은 어디인지 확인하기
df_egg30[df_egg30['가격(원)'] == 0]

# 최소 가격 제외하기
df_egg30 = df_egg30[df_egg30['가격(원)'] != 0]

# 다시 최대, 최소, 평균 확인
df_egg30['가격(원)'].agg(['mean', 'max', 'min', 'median'])

# 최대 확인
df_egg30[df_egg30['가격(원)'] == 75000]

# 원하는 구의 달걀 가격 확인하기
gu = input('알고 싶은 구 : ')
condition = (df_egg30['자치구 이름'] == gu)
df_egg30_gu = df_egg30[condition]

df_egg30_gu.head(1)

# 광진구의 달걀 평균 가격, 최대, 최소 알아보기
df_egg30_gu['가격(원)'].agg(['mean', 'max', 'min', 'mean'])

# +
# 시각화
x = df_egg30_gu['시장/마트 이름']
y = df_egg30_gu['가격(원)']

plt.scatter(x, y)
plt.title(f'{gu} 달걀(30개) 가격')
plt.grid(True)
plt.show()
# -

# 마트별 달걀 가격
mart = input('시장/마트 이름 : ')
condition = df_egg30['시장/마트 이름'].str.contains(mart)
df_egg30_mart = df_egg30[condition]

# +
# 시각화
x = df_egg30_mart['시장/마트 이름']
y = df_egg30_mart['가격(원)']

fig, ax = plt.subplots(1, 2)

ax[0].boxplot(y)

ax[1].scatter(x, y)
plt.title('이마트별 달걀(30개) 가격')
plt.grid(True)
plt.xticks(rotation=60)

plt.tight_layout()
plt.show()
# -

# 만약 추가 분석을 원하면..?
# 함수를 만들어서 계속 돌려가며 찾으면 된다.























