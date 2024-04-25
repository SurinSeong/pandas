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

# # Numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# * numpy의 array : 단일 데이터 타입이다.
#     * 정수와 실수가 혼용되어 있으면 실수로 변경된다.

print(np.array([30, 10.54, 56.7, 180, 73]))

# * 데이터 타입 지정 옵션 : dtype=___
# * 데이터 타입 변경 : astype()

# +
# np.array 생성
print(np.array([[12, 21],
                [46, 65]]))

# 데이터 타입 지정 (float)
print(np.array([[12, 21],
                [46, 65]], dtype=float))
# -

arr = np.arange(10, 100, 10)
print(arr)
print(arr.dtype)
# 데이터 타입 변경
print(arr.astype(float))

# np.zeros()
print(np.zeros(10, dtype=int))
# np.ones()
print(np.ones((3, 5), dtype=float))

# np.arange(start, end, step)
np.arange(0, 20, 3)

# np.linspace()
np.linspace(-10, 11, 4)

# 난수 활용 -> 배열 만들기
print(np.random.random((2, 5)))
print(np.random.rand(2, 5))
print(np.random.normal(10, 1, (2, 4)))
print(np.random.randn(2, 4))
print(np.random.randint(1, 20, (3, 2)))

# 배열의 기초
x = np.random.randint(100, size=(2, 4))
print(x)
print(x.ndim)
print(x.shape)
print(x.size)
print(x.dtype)

# * indexing

a = np.arange(10)
print(a)
print(a[4])
print(a[-2])

# 인덱싱을 이용해 값 변경
a[0] = -1
print(a)

# * 슬라이싱

print(a)
print(a[:4])
print(a[-3:])
print(a[::-2])
print(a[::3])

# * reshape : 차원 변경

# +
x = np.arange(20)
print(x)
print(x.shape)

x2 = x.reshape(4, -1)
print(x2)
print(x2.shape)
# -

# concatenate : 배열 이어붙이기
x1 = np.array([10, 100, 1000])
x2 = np.array([5, 0, -5])
print(x1)
print(x2)
print(np.concatenate([x1, x2]))
# # 1차원에서는 행, 열 방향이 없음 !!!
# print(np.concatenate([x1, x2], axis=1))

# 2차원 이상의 배열에서 concat으로 이어붙이기
# 축 변경하면서
m = np.arange(9).reshape(3, 3)
print(m)
print(np.concatenate([m, m]))
print(np.concatenate([m, m], axis=1))

h = np.arange(25).reshape(5, -1)
print(h)
print(np.split(h, [3], axis=0))
print(np.split(h, [3], axis=1))

h1, h2 = np.split(h, [3], axis=0)
print(h1)
print(h2)


# ## Numpy 연산

# array의 모든 원소에 10을 더하는 함수 생성
# for 반복문 이용
def add_10(values):
    output = np.empty(len(values))

    for i in range(len(values)):
        output[i] = values[i] + 10

    return output


nums = np.random.randint(1, 100, size=10)
print(nums)
print(add_10(nums))

# numpy 특성 이용
big_arr = np.random.randint(-100, 100, size=20)
print(big_arr)
print(big_arr + 10)

## np.empty()
np.empty(5)
# 5개의 원소를 가진 1차원 배열 생성
# 초기화는 되어있지 않음. > 여기에 출력되는 값은 메모리에 이미 존재한 임의의 값으로 배열을 채움.

# 기본 연산
x = np.arange(10)
print(x + 15)
print(x - 15)
print(x * 15)
print(x / 15)


# +
# 행렬 간 연산
x = np.arange(6).reshape(2, 3)
print(x)

y = np.random.randint(100, size=(2, 3))
print(y)
# -

print(x + y)
print(x - y)
print(x * y)
print(x / y)

# Broadcasting
print(x)
print(x + (-1))

# 집계함수
x = np.arange(10).reshape(2, 5)
print(x)
print(np.sum(x))
print(np.mean(x))
print(np.min(x))
print(np.max(x))

# axis 사용해서 집계함수
print(x)
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

x = np.arange(10)
print(x < 3)
print(x > 8)

# 특정 정보를 추출하고 싶을 때
x[x > 8]

# 실무에서..
condition = (x > 8)
x[condition]

# # Pandas Basic

# +
# Series : index를 달고 출력
s = pd.Series([-5, -4, -3, -2, -1])
print(s)

# index명, Series명 설정해주는 것도 가능
data = pd.Series([-5, -4, -3, -2, -1], index=['a', 'b', 'c', 'd', 'e'], name='Title')
print(data)

# index명으로 출력 가능
print(data['c'])

# +
# Dictionary >> Series
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
    'USA' : 2041280000
}
print(gdp)

sr_gdp = pd.Series(gdp)
print(sr_gdp)

country = \
pd.DataFrame(
    {
        'population' : sr_population,
        'GDP' : sr_gdp
    }
)
country
# -

# 연산자 활용
print(country['GDP'])
print(country['population'])

# 1인당 국민 총 생산량
gdp_per_capita = country['GDP'] / country['population']
gdp_per_capita

# 1인당 국민 총 생산을 country에 넣기
country['GDP per capita'] = gdp_per_capita
country

# ## indexing / slicing
# * .loc[행, 열] : 명시적
# * .iloc[행, 열] : 비명시적 (인덱스 접근)

country

country.loc['China']

country.loc['Korea':'Japan', :]

country.loc['Japan':'China', :'GDP']

country.iloc[0]

country.iloc[1:3, :-1]

# DataFrame에 새로운 데이터 추가 및 수정
# 추가 : 리스트 or 딕셔너리 추가
df = pd.DataFrame(columns=['Name', 'Age', 'Address'])
df.loc[0] = ['Alice', 26, 'Yangyang']
df.loc[1] = ['Paul', 24, 'Hadong']
df.loc[2] = {'Name' : 'Sue', 'Age' : 28, 'Address' : 'Andong'}
df

df.loc[1, 'Name']

df.loc[2, 'Name'] = 'Nick'
df

# 새로운 컬럼 추가
df['tel'] = None
df.loc[0, 'tel'] = '010-0000-0000'
df

df['Name']

df[['tel', 'Name']]

# 결측치 처리
# 누락 데이터 처리
df.isnull()

print(df)
print(df.isnull())

df.dropna()

df2 = df.copy()
df2

# 결측치 채우기
# 채우고 업데이트 할 때 해당 컬럼명을 꼭 작성 !
df2['tel'] = df2['tel'].fillna(0)
df2

# ## Series 연산

a = pd.Series([2, 4, 6], index=[0, 1, 2])
b = pd.Series([1, 3, 5], index=[1, 2, 3])
print(a)
print(b)

a + b

a.add(b, fill_value=0)

c = pd.DataFrame(np.random.randint(0, 10, (2, 2)), columns=['A', 'B'])
d = pd.DataFrame(np.random.randint(0, 10, (3, 3)), columns=['B', 'A', 'C'])
c * d

c.add(d, fill_value=0)

# 집계함수
data = {
    'a' : [(i + 5) for i in range(3)],
    'b' : [(i ** 2) for i in range(3)]
}
data

df = pd.DataFrame(data)
df

df['a'], df['a'].sum()

df.sum(), df.mean()

df = \
pd.DataFrame(
    {
        'col1' : [2, 1, 9, 8, 7, 4],
        'col2' : ['a', 'a', 'b', np.nan, 'd', 'c'],
        'col3' : [0, 1, 9, 4, 2, 3]
    }
)
df.head()

df.sort_values('col1')

df.sort_values('col1', ascending=False)

df.sort_values(['col2', 'col1'])

df.sort_values(['col1', 'col2'], ascending=[True, False])

# # Pandas Advanced

df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
df

df['B'] < 0.5

df[(df['B'] < 0.5) & (df['A'] > 0.3)]

# 조건을 변수에 넣어서 원하는 데이터 확인한다.
condition = (df['B'] < 0.5) & (df['A'] > 0.3)
df[condition]

# df.query() 사용
df.query('A < 0.5 and B > 0.3')

# +
# 문자열 조건 검색
data = {
    'Animal' : ['Dog', 'Cat', 'Cat', 'Pig', 'Cat'],
    'Name' : ['Happy', 'Sam', 'Tom', 'Mini', 'Rocky']
}

# 데이터 프레임 만들기
df = pd.DataFrame(data)
df.head(3)
# -

df['Name']

cat = df['Animal'].str.contains('Cat')

df[cat]

# ## 함수로 데이터 처리

df = pd.DataFrame(np.arange(10), columns=['num'])
df


# +
# 제곰함수 만들기
def square(x):
    return x ** 2

df['num'], df['num'].apply(square)
# -

df['square'] = df['num'].apply(square)
df

df['square_lambda'] = df.num.apply(lambda x : x ** 2)
df

# +
# 또다른 예시
df = pd.DataFrame(columns=['apt_number'])

# 데이터 삽입
df.loc[0] = '333동 601호'
df.loc[1] = '333-육공일'
df.loc[2] = '삼삼삼동 육공일호'
df.loc[3] = '삼3삼-6공일'

# 새로운 컬럼
df['apt_number_new'] = ''
df


# -

# 전처리 함수 만들기
def preprocess(apt_number):
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
        '구' : '9',
        '동' : '-',
        '호' : '',
        ' ' : ''
    }
    for k, v in mapping.items():
        apt_number = apt_number.replace(k, v)
    return apt_number


df.apt_number_new = df.apt_number.apply(preprocess)
df

# ## 그룹

data = {
    'group' : ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'x' : [12, 23, 65, 78, 23, 65, 89, 42],
    'y' : [1, 2, 3, 4, 5, 6, 7, 8]
}
df = pd.DataFrame(data)
df.head(3)

df.groupby('group').agg(['min', 'max'])

df.groupby('group').agg({'x' : 'mean', 'y' : 'sum'})












