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

# +
# Series : index를 달고 출력됨
s = pd.Series([1, 2, 3, 4])
print(s)

# index명 설정해주는 것도 가능함.
data = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'], name='Title') # name : Series명 만들어줌.
print(data)

# index명으로 접근해 값 출력 가능
print(data['b'])

# +
# Dictionary(key-value) >> Series(index-value)
population = {
    'Korea' : 5180,
    'Japan' : 12718,
    'China' : 141500,
    'USA' : 35676
}
print(population)

sr_population = pd.Series(population)
print(sr_population)

# +
# DataFrame
gdp = {
    'Korea' : 169320000,
    'Japan' : 516700000,
    'China' : 1409250000,
    'USA' : 2041280000,
}
print(gdp)

sr_gdp = pd.Series(gdp)
print(sr_gdp)

country = \
pd.DataFrame(
    {
        'population' : sr_population, 'GDP' : sr_gdp    
    }
)
country

# +
# 연산자 활용
print(country['GDP'])
print(country['population'])

# 1인당 국민 총 생산량
gdp_per_capita = country['GDP'] / country['population']
gdp_per_capita
# -

# 1인당 국민 총 생산을 country에 삽입하기
print(country)
country['GDP per capita'] = gdp_per_capita
print(country)

# 자료 저장
# CSV : Comma Separated Value
country.to_csv('./country.csv')

# Excel
country.to_excel('./country.xlsx')

# 불러오기
pd.read_csv('./country.csv')

# +
# indexing
# .loc[행, 열] : 명시적
# .iloc[행, 열] : 비명시적 (인덱스 접근)

country
# -

country.loc['China'] # Series로 출력됨

country.loc['Korea':'Japan', :'population'] # [a:b] 구간의 모든 값 포함. a부터 b까지

country.loc['Korea':'Japan', :'GDP']

country.iloc[0] # 인덱스가 0인 Korea의 값만 출력

country.iloc[1:3, :-1]

# +
# DataFrame에 새로운 데이터 추가 및 수정
# 추가 : 리스트 or 딕셔너리 추가

df = pd.DataFrame(columns=['Name', 'Age', 'Address'])
df.loc[0] = ['남학균', 26, '서울']
df.loc[1] = {'Name' : '성수린', 'Age' : 25, 'Address' : '제주'}
df
# -

df.loc[1, 'Name']

# 데이터 변경
df.loc[1, 'Name'] = '아이린'
df

# 새로운 컬럼 추가
df['tel'] = None
df.loc[0, 'tel'] = '010-1234-5678'
df

df['Name']

# 출력 순서를 정할 수 있음.
# 여러 컬럼을 확인할 때는 []으로 감싸준다.
df[['Age', 'Name']]

# +
# 결측치 처리
# 누락 데이터 처리
df.isnull() # bool값으로 알려줌

print(df)
print(df.isnull())
# -

df.dropna() # 결측치 삭제 --> update를 해야 저장이 됨.

# 데이터프레임 복사하기 --> 원본데이터 훼손 방자
df2 = df.copy()
df2

# 결측치 채우기
# 채우고 업데이트 할 때 해당 컬럼명을 꼭 작성해야함.
df2['tel'] = df2['tel'].fillna(0) # --> update 해야 함.
print(df2)

# Series 연산
a = pd.Series([2, 4, 6], index=[0, 1, 2])
b = pd.Series([1, 3, 5], index=[1, 2, 3])
print(a)
print(b)

a + b # 각 series의 인덱스가 없는 것이 있기 때문

a.add(b, fill_value=0) # 결측치 값을 0으로 바꾼 후 연산

# +
# 수업 내용 다시 참고해서 해보기..

c = pd.DataFrame(np.random.randint(0, 10, (2, 2)), columns=['A', 'B'])
d = pd.DataFrame(np.random.randint(0, 10, (3, 3)), columns=['B', 'A', 'C'])
c + d
# -

c.add(d, fill_value=0)

# +
# 집계함수

data = {
    'a' : [(i + 5) for i in range(3)],
    'b' : [(i ** 2) for i in range(3)]
}
data
# -

# dataframe에 넣어주기
df = pd.DataFrame(data)
df

df['a']

df['a'].sum()

# 전체 합계
df.sum()

# 전체 평균
df.mean()

# 값으로 정렬
df = \
pd.DataFrame(
    {
        'col1' : [2, 1, 9, 8, 7, 4],
        'col2' : ['a', 'a', 'b', np.nan, 'd', 'c'],
        'col3' : [0, 1, 9, 4, 2, 3]
    }
)
df.head()

# sort_values() : default=ASC
# 오름차순이 기본, 'col1'인 컬럼을 기준으로 정렬
df.sort_values('col1')

# 내림차순 정렬
df.sort_values('col1', ascending=False)

# 여러 컬럼으로 정렬할 경우, 먼저 오는 컬럼이 우선
# 여러 컬럼이면 []로 묶어준다.
df.sort_values(['col2', 'col1'])

# Q. 여러 컬럼으로 정렬할 경우, 오름차순, 내림차순을 다르게 적용할 수 있나?
df.sort_values(['col1', 'col2'], ascending=[True, False]) # 앞의 컬럼 넣은 것 처럼 ascending에 리스트 형태로 넣어주면 된다.


