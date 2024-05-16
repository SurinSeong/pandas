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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# %matplotlib inline

# 그래프에서 마이너스 폰트 깨지는 문제 대처
mpl.rcParams['axes.unicode_minus'] = False

# 폰트 확인하기
print([f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name])
print([f.name for f in fm.fontManager.ttflist if 'Malgun' in f.name])

# 폰트 지정
plt.rcParams['font.family'] = 'NanumGothic'
# -

# 그래프 그려보기
plt.plot([1, 2 ,3], [4, 5, 6]) # 선그래프
plt.title('Graph')
plt.show()

# # 코로나19 데이터 분석

# ## 필요한 데이터 불러오기

# 데이터 불러와서 변수에 저장하고 복사해서 사용하기
raw = pd.read_csv('../datasets/서울시 코로나19 확진자 현황.csv')
df = raw.copy()

# ## 전처리

# 데이터 확인하기
df.head(2)

# 컬럼명 확인
df.columns

# 컬럼 확인하기
df['이동경로'].unique()

# 불필요한 데이터 or 컬럼 삭제
df.drop(columns=['환자번호', '국적', '환자정보', '조치사항', '이동경로', '등록일', '수정일', '노출여부'], inplace=True) # df = 해서 수정할 필요없이 바로 삭제함.

# 삭제된 것 확인하기
df.head(1)

# 자료형 변환
df.dtypes

# '확진일' 컬럼의 타입 변경하기
df['확진일'] = pd.to_datetime(df['확진일'])

# 변경된 것 확인하기
df.dtypes

# +
# '지역' 컬럼 확인
print(df['지역'].nunique(), df['지역'].unique())

# 문자 공백 제거하고 수정하기
df['지역'] = df['지역'].str.strip()
# -

# 제거 되었는지 확인하기
print(df['지역'].nunique(), df['지역'].unique())

# 지역 type 변경
df['지역'] = df['지역'].astype('category')
df.dtypes

# 전체적인 df의 정보 확인
df.info()

# 결측치 제거
df.isnull().sum()

# 구별 확진자 동향 : 확진일-구별 확진자 수 집계
# 피벗테이블 만들고 변수 저장
# 날짜 순서대로 지역별로 확인 가능함.
df_gu = \
pd.pivot_table(df, index='확진일', columns='지역', values='연번', aggfunc='count', margins=True)

# 서울시 일별 추가 확진자 동향 파악
# df_gu의 마지막 row는 누적 합이기 때문에 없애도 된다.
s_date = df_gu['All'][:-1]

# 서울시 일별 추가확진자가 많았던 순서로 확인
s_date.sort_values(ascending=False)

# +
# 서울시 일별 추가 확진자 시각화
x, y = s_date.index, s_date.values # 일자별, 확진자수

# 그래프 그리기
plt.plot(x, y)
plt.title('서울시 일별 추가 확진자 (2021.09.28 현재)')
plt.xlabel('확진일')
plt.ylabel('추가 확진자 수')
plt.xticks(rotation=45)

# 그래프 보여주기
plt.show()

# +
# df_gu
# -

# 서울시 구별 누적 확진자 비교
s_gu = df_gu.loc['All'][:-1]

# 서울시 구별 누적 확진자가 많은 순으로 보기
s_gu = s_gu.sort_values(ascending=False)

# +
# 서울시 구별 누적 확진자 시각화
x, y = s_gu.index, s_gu.values

# 배경 만들기
plt.figure(figsize=(10, 6))
plt.title('서울시 구별 누적 확진자 (2021.09.28 현재)', size=20)

# 그래프 그리기
plt.barh(x, y)

# 그래프 보여주기
plt.show()
# -

# 최근일 기준, 지역별 추가 확진자 현황 파악
s_gu_lateset = df_gu.iloc[-2, :-1].sort_values(ascending=False)

# 시각화
print(s_gu_lateset)
x, y = s_gu_lateset.index, s_gu_lateset.values
plt.figure(figsize=(10, 6))
plt.barh(x, y)
plt.show()

# +
# 접촉력에 따른 확진 건수 BEST 10 선정
print(df['접촉력'])

# unique한 값들 세어서 내림차순 정렬 후, BEST 10 뽑고 DF로 만들기
df['접촉력'].value_counts()[:10].to_frame()

# +
# 최근 월 접촉력에 따른 확진 건수 BEST 10 
# 최근 월 : 2021-09
# 조건 설정하기
condition = (df['확진일'].dt.year == 2021) & (df['확진일'].dt.month == 9)

# 조건을 이용해 최근 월 뽑기
print(df[condition])

# 조건의 접촉력만 뽑고 접촉력 BEST 10 확인하고 DF로 바꿔서 확인하기
df[condition]['접촉력'].value_counts()[:10].to_frame()
# -












