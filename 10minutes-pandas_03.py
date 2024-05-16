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

# * isna() : 결측치가 존재하니? (존재하면 True, 아니면 False)

print(pd.isna('dog'))
print(pd.isna(pd.NA))
print(pd.isna(np.nan))

arr = np.array([[1, np.nan, 3],
                [4, 5, np.nan]])
print(arr)
print(pd.isna(arr))

index = pd.DatetimeIndex(['2017-07-05', '2017-07-06', None, '2017-07-08'])
print(index)
print(pd.isna(index))

df = pd.DataFrame([['ant', 'bee', 'cat'],
                   ['dog', None, 'fly']])
print(df)
print(pd.isna(df))
print(pd.isna(df[1]))

# ## Flexible binary operations(연산)
# * Broadcasting
# * missing data

# +
# 데이터프레임 설정
df = pd.DataFrame(
    {
        'one' : pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
        'two' : pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
        'three' : pd.Series(np.random.randn(3), index=['b', 'c', 'd']),
    }
)
print(df)

# 인덱스가 1인 row 추출
row = df.iloc[1]
# 인덱스명이 two인 column 추출
column = df['two']

# df에서 변수 row를 뺀 df를 column을 기준으로 추출
print(df.sub(row, axis='columns'))
print(df.sub(row, axis=1))

# df에서 변수 column을 뺀 df를 row를 기준으로 추출
print(df.sub(column, axis='index'))
print(df.sub(column, axis=0))
# -

# * sub() : 해당 row나 column빼고 출력
#     * level : 
# int or labe
#     * --> 
# Broadcast across a level, matching Index values on the passed MultiIndex level.

# +
# df를 복사해서 새로운 변수에 저장하기
dfmi = df.copy()

# dfmi의 인덱스를 새롭게 설정하기
dfmi.index = pd.MultiIndex.from_tuples(
    [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'a')], names=['first', 'second']
)

# 위에서 지정한 column을 뺀 dfmi 출력
print(dfmi.sub(column, axis=0, level='second'))
# -

# * divmod() : 몫과 나머지를 출력

# +
# series 만들기
s = pd.Series(np.arange(10))
print(s)

div, rem = divmod(s, 3)
print(div)
print(rem)

# +
# 인덱스를 지정해준다.
idx = pd.Index(np.arange(10))
print(idx)

# 지정해준 인덱스를 나눠준다.
div, rem = divmod(idx, 3)
print(div, rem)
# -

# 나누는 수를 리스트로 지정할 수 있음.
div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
print(div)
print(rem)

# +
# missing data --> operations with fill values
df2 = df.copy()

# 결측치를 추가해줌
df2.loc['a', 'three'] = 1.0
print(df)
print(df2)
print(df + df2) # 결측치 + 숫자 = 결측치
print(df.add(df2, fill_value=0))
# -

# ### stats
# * 일반적인 연산에서는 결측치를 배제한다.

print(df)
# column별 평균
print(df.mean())
# row별 평균
print(df.mean(axis=1))

# +
# index를 dates로 하고 세 번째 인덱스부터 값 넣기
dates = pd.date_range('20130101', periods=6)
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print(s)

# s를 뺀 df를 출력
print(df.sub(s, axis='index'))
# -

# ## User defined functions
# * DataFrame.agg() / DataFrame.transform()

df.agg(lambda x : np.mean(x) * 5.6)

df = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [np.nan, np.nan, np.nan]],
                  columns=['A', 'B', 'C'])
print(df.agg(['sum', 'min']))
print(df.agg({'A' : ['sum', 'min'],
              'B' : ['min', 'max']}))
print(df.agg(x = ('A', 'max'),
             y = ('B', 'min'),
             z = ('C', 'mean')))
print(df.agg('mean', axis='columns'))

df.transform(lambda x : x * 101.2)

df = pd.DataFrame({'A' : range(3),
                   'B' : range(1, 4)})
print(df)
print(df.transform(lambda x : x + 1))

s = pd.Series(range(3))
print(s)
print(s.transform([np.sqrt, np.exp]))

df = pd.DataFrame({
    'Date' : [
        "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05",
        "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05"],
    'Data' : [5, 8, 6, 1, 50, 100, 60, 120]
})
print(df)
print(df.groupby('Date')['Data'].transform('sum'))

# +
df = pd.DataFrame({
    'c' : [1, 1, 1, 2, 2, 2, 2],
    'type' : ['m', 'n', 'o', 'm', 'm', 'n', 'n']
})
print(df)

df['size'] = df.groupby('c')['type'].transform(len)
print(df)
# -

# ## Value Counts

s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())

# ## String Methods

s = pd.Series(['A', 'B', 'C', 'AaBa', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())

# # Merge

# ## Concat

df = pd.DataFrame(np.random.randn(10, 4))
df

# break it into pieces
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# combine two Series
s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])
print(s1)
print(s2)
print(pd.concat([s1, s2]))

# 원래의 인덱스 없애고 합치고 새로운 인덱스 설정하기
print(pd.concat([s1, s2], ignore_index=True))

# 합친거 보여주기
pd.concat([s1, s2]. keys=['s1', 's2'])

df1 = pd.DataFrame([['a', 1], ['b', 2]],
                  columns=['letter', 'number'])
df1

df2 = pd.DataFrame([['c', 3], ['d', 4]],
                  columns=['letter', 'number'])
df2

pd.concat([df1, df2])

df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
                  columns=['letter', 'number', 'animal'])
df3

pd.concat([df1, df3], sort=False)

# inner join
pd.concat([df1, df3], join='inner')

df4 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']],
                  columns=['animal', 'name'])
df4

# 옆으로 붙이기
pd.concat([df1, df4], axis=1)

df5 = pd.DataFrame([1], index=['a'])
df5

df6 = pd.DataFrame([2], index=['a'])
df6

# +
# # ValueError: Indexes have overlapping values: Index(['a'], dtype='object')
# pd.concat([df5, df6], verify_integrity=True)
# -

df7 = pd.DataFrame({'a' : 1, 'b' : 2}, index=[0])
df7

new_row = pd.Series({'a' : 3, 'b' : 4})
new_row

pd.concat([df7, new_row.to_frame().T], ignore_index=True)

# ## Join

left = pd.DataFrame({'key' : ['foo', 'foo'], 'lval' : [1, 2]})
right = pd.DataFrame({'key' : ['foo', 'foo'], 'rval' : [4, 5]})
left, right

pd.merge(left, right, on='key')

left = pd.DataFrame({'key' : ['foo', 'bar'], 'lval' : [1, 2]})
right = pd.DataFrame({'key' : ['foo', 'bar'], 'rval' : [4, 5]})
left, right

pd.merge(left, right, on='key')

# * merge()

df1 = pd.DataFrame({'lkey' : ['foo', 'bar', 'baz', 'foo'],
                    'value' : [1, 2, 3, 5]})
df2 = pd.DataFrame({'rkey' : ['foo', 'bar', 'baz', 'foo'],
                    'value' : [5, 6, 7, 8]})
df1, df2

df1.merge(df2, left_on='lkey', right_on='rkey')

df1.merge(df2, left_on='lkey', right_on='rkey',
          suffixes=('_left', '_right'))

df1 = pd.DataFrame({'a' : ['foo', 'bar'], 'b' : [1, 2]})
df2 = pd.DataFrame({'a' : ['foo', 'baz'], 'c' : [3, 4]})
df1, df2

df1.merge(df2, how='inner', on='a')

df1.merge(df2, how='left', on='a')

df1 = pd.DataFrame({'left' : ['foo', 'bar']})
df2 = pd.DataFrame({'right' : [7, 8]})
df1, df2

df1.merge(df2, how='cross')

# # grouping
# * splitting
# * applying
# * combining

df = pd.DataFrame(
    {
        'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C' : np.random.randn(8),
        'D' : np.random.randn(8),
    }
)
df

# A를 기준으로 묶고 C와 D의 합계 구하기
df.groupby('A')[['C', 'D']].sum()

df.groupby(['A', 'B']).sum()

speeds = pd.DataFrame(
    [
        ('bird', 'Falconiformes', 389.0),
        ('bird', 'Psittaciformes', 24.0),
        ('mammal', 'Carnivora', 80.2),
        ('mammal', 'Primates', np.nan),
        ('mammal', 'Carnivora', 58),
    ],
    index=['falcon', 'parrot', 'lion', 'monkey', 'leopard'],
    columns=('class', 'order', 'max_speed'),
)
speeds

grouped=speeds.groupby('class')

df = pd.DataFrame(
    {
        'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C' : np.random.randn(8),
        'D' : np.random.randn(8),
    }
)
df

# ### groupby()

df = pd.DataFrame({'Animal' : ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                   'Max Speed' : [380., 370., 24., 26.]})
df

df.groupby(['Animal']).mean()

arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
          ['Captive', 'Wild', 'Captive', 'Wild']]
index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
df = pd.DataFrame({'Max Speed' : [390., 350., 30., 20.]},
                  index=index)
df

df.groupby(level=0).mean()

df.groupby(level='Type').mean()

l = [[1, 2, 3],
     [1, None, 4],
     [2, 1, 3],
     [1, 2, 2]]
df = pd.DataFrame(l, columns=['a', 'b', 'c'])
df

df.groupby(by=['b']).sum()

df.groupby(by=['b'], dropna=False).sum()

l = [['a', 12, 12],
     [None, 12.3, 33.],
     ['b', 12.3, 123],
     ['a', 1, 1]]
df = pd.DataFrame(l, columns=['a', 'b', 'c'])
df

df.groupby(by='a').sum()

df.groupby(by='a', dropna=False).sum()

df = pd.DataFrame({'Animal' : ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
                   'Max Speed' : [380., 370., 24., 26.]})
df.groupby('Animal', group_keys=True)[['Max Speed']].apply(lambda x : x)










