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
import seaborn as sns

# # visualization

# +
path = '../../datasets/raw_sales.csv'

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
# plt.savefig('../data/average_of_YP.png')

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
len(result['year'].unique())

# +
N = int(len(result['year'].unique()))

fig, ax = plt.subplots(nrows=N, ncols=1, figsize=(6, 3*N))

for i, y in enumerate(result['year'].unique()):
    print(i, y)

    data = result.loc[result['year'] == y]
    ax[i].bar(data['month'], data['price'])
    ax[i].set_title(f'year{y}')

plt.tight_layout()
plt.show()
# -

# ## Seaborn

# +
fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(data=df, x='month', y='price', ax=ax)
ax.set_title('Average of House Price per Year with sns', size=16)
ax.set_xlabel('Year', size=12)
ax.set_ylabel('Price', rotation=0, size=12)

plt.show()
# -


# ## 이상치 확인하고 조작하기

# boxplot / outlier (이상치 제거)
# 2007 ~ 2010 추출
# raw 확인
raw.tail(3)
# df 확인하기
df.tail(3)

sales = df.copy()
sales['year'].unique()

condition = sales.year.isin([2007, 2008, 2009, 2010])
sales_condition = sales.loc[condition, :].reset_index(drop=True)
sales_condition.tail(3)

# +
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=sales_condition, x='year', y='price', hue='propertyType')

plt.show()
# -

# boxplot 개념
dat = {'values' : [1, 12, 12, 13, 12, 11, 14, 13, 15, 102, 120, 14, 14, 17, 18, 19, 20]}
temp_df = pd.DataFrame(dat)
temp_df[:5]

# +
# 1단계 : 사분위수 (Q1, Q3, IQR)
Q1 = temp_df['values'].quantile(0.25)
Q3 = temp_df['values'].quantile(0.75)
IQR = Q3 - Q1

print(Q1)
print(Q3)
print(IQR)
# -

# 2단계 : 이삼치 임계값(threshold) 설정
lower_bound = Q1 - 1.5 * IQR # 하한가
upper_bound = Q3 + 1.5 * IQR # 상한가
print(lower_bound)
print(upper_bound)

condition = (temp_df.values < lower_bound) | (temp_df.values > upper_bound)
condition.sum()

# 3단계 : 이상치 확인
outlier = temp_df.loc[condition, :]
outlier

# 4단계 : 이상치 처리 (제거, bound 흡수)
# 1) 제거
temp_clean = temp_df.loc[~condition, :]
temp_clean

# 2) 이상치를 임계값으로 넣기
# min(a, b) : 더 작은거 반환, max(a, b) : 더 큰 거 반환
# apply 적용
temp_df['values'] = temp_df['values'].apply(lambda x : min(x,  upper_bound))
print(temp_df)
temp_df['values'] = temp_df['values'].apply(lambda x : max(x,  lower_bound))
temp_df

# 변수
dat = {'values' : [1, 12, 12, 13, 12, 11, 14, 13, 15, 102, 120, 14, 14, 17, 18, 19, 20]}
temp_df = pd.DataFrame(dat)
temp_df[:5]


# 함수로 만들기
# 1. 이상치 제거 함수
def remove_outlier(df, column):
    # 1단계 : 사분위수 가져오기
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # 임계값 가져오기
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 이상치 구분하기
    condition = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier = df.loc[condition, :].reset_index(drop=True)
    temp_clean = df.loc[~condition, :].reset_index(drop=True)

    return outlier, temp_clean


remove_outlier(temp_df, 'values')

# ### 실무

# 실제 DF에 적용해보기
# sales_condition
sales_condition

# 아까 결측치있는 것을 boxplot으로 봤기 때문에 year별 결측치를 제거하는 것을 한다.
sales_condition_outlier, sales_condition_clean = remove_outlier(sales_condition, 'price')

sales_condition_clean

# * 시각화

# 2007 ~ 2010 추출 결과 데이터 시각화
# 과학적 표기법 비활성 설정
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter()
formatter.set_scientific(False)

# +
fig, ax = plt.subplots(nrows=2, figsize=(10, 12))

sns.boxplot(data=sales_condition, x='year', y='price', hue='propertyType', ax=ax[0])
ax[0].set_title('WITH outliers')
ax[0].yaxis.set_major_formatter(formatter)

sns.boxplot(data=sales_condition_clean, x='year', y='price', hue='propertyType', ax=ax[1])
ax[1].set_title('WITHOUT outliers')
ax[1].yaxis.set_major_formatter(formatter)

plt.show()
# -





# 함수 2
# 이상치 변경 함수 만들기
def change_outlier(df, column):
    # 1단계 : 사분위수 가져오기
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # 임계값 가져오기
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 이상치 구분하기
    condition = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier = df.loc[condition, :]
    # 이상치를 임계값에 넣어주기
    df[column] = df[column].apply(lambda x : min(x,  upper_bound))
    df[column] = df[column].apply(lambda x : max(x,  lower_bound))
    # 변경된 값으로 된 DF를 새로운 변수에 저장하기
    change_df = df.reset_index(drop=True)
    
    return change_df


# +
# change_outlier(temp_df, 'values')
# -

# sales_condition 이상치 변경
sales_condition_change = change_outlier(sales_condition, 'price')
sales_condition_change

# +
fig, ax = plt.subplots(nrows=2, figsize=(10, 12))

sns.boxplot(data=sales_condition, x='year', y='price', hue='propertyType', ax=ax[0])
ax[0].set_title('WITH outliers')
ax[0].yaxis.set_major_formatter(formatter)

sns.boxplot(data=sales_condition_change, x='year', y='price', hue='propertyType', ax=ax[1])
ax[1].set_title('CHANGE outliers')
ax[1].yaxis.set_major_formatter(formatter)

plt.show()
# -

# ## subplots
# * 다수의 그래프를 하나의 창에 나누어 그리는데 사용
# * 개별적으로 조절 가능한 축 >> 'ax'

tips = sns.load_dataset('tips')
tips.head(3)

# +
# 그래프 그리기
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
# 요일별 총 청구금액 시각화
sns.barplot(x='day', y='total_bill', data=tips, ax=ax[0,0])
ax[0,0].set_title('total_bill per day')

# 요일별
sns.countplot(x='day', data=tips, ax=ax[0, 1])
ax[0, 1].set_title('tips per day')

sns.stripplot(x='day', y='total_bill', data=tips, ax=ax[1, 0])
ax[1, 0].set_title('stripplot of tips')

sns.pointplot(x='day', y='total_bill', data=tips, ax=ax[1, 1])
ax[1, 1].set_title('pointplot of tips')

plt.tight_layout()
plt.show()

# +
# 설정: 2행 2열의 서브플롯 생성
nrows, ncols = 2, 2

# 서브플롯 생성
fig, axs = plt.subplots(nrows, ncols, figsize=(6, 6))

# 각 서브플롯에 인덱스 표시
for i in range(nrows):
    for j in range(ncols):
        axs[i, j].text(0.5, 0.5, f'Index {i}, {j}',
                       ha='center', va='center', fontsize=12)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        
plt.tight_layout()
# plt.savefig("output/subplots.png")
plt.show()
# -

# ## ploty

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'jupyterlab'

df = px.data.tips()
fig = px.box(df, y="total_bill")
fig.show()

# +
import plotly.graph_objects as go # 저수준 그래프

# 가상의 데이터 생성
x = np.arange(0, 15, 1)

y1, y2 = x**2, x**3

fig = go.Figure()

# 시각화 차트
# 지정한 x, y 값을 plotly를 이용해서 
fig.add_trace(go.Scatter(x=x, y=y1))
fig.add_trace(go.Scatter(x=x, y=y2))
fig.add_trace

fig.update_layout(
    title = 'basic graph',
    xaxis_title = 'x축',
    yaxis_title = 'y축',
    template = 'plotly_white'
)
# 그래프 보여주기
fig.show()

# +
# 가상의 데이터 생성
x = np.arange(0, 15, 1)

y1, y2 = x**2, x**3

# fig 생성
fig = go.Figure()

# 시각화 차트
# 지정한 x, y 값을 plotly를 이용해서 
for y in [y1, y2]:
    fig.add_trace(go.Scatter(x=x, y=y))

fig.update_layout(
    title = 'basic graph',
    xaxis_title = 'x축',
    yaxis_title = 'y축',
    template = 'plotly_white'
)
# 그래프 보여주기
fig.show()

# +
# 샘플데이터 만들기
data = {
    'x' : ['a', 'b', 'c', 'd'],
    'y1' : [10, 15, 13, 17],
    'y2' : [16, 8, 13, 10]
}

# 기본 차트
fig = px.bar(data, x='x', y='y1', title='basic chart')

# 차트 추가
fig.add_trace(go.Scatter(x=data['x'], y=data['y2'],
                         mode='lines+markers', name='Line Chart'))

# 막대차트
# 업데이트 type이 산점도 : selector=dict(type='scatter')
fig.update_traces(marker_color='lightblue', selector=dict(type='scatter'))

# 그래프 보여주기
fig.show()


# +
# 선그래프 line graph

# +
# sales 이용
# -

sales.columns

sales['datesold'].dt.year

# +
# 2008 vs. 2018
condition = sales['year'].isin([2008, 2018])

df = sales[condition]
result = round(df.groupby(['year', 'month'])['price'].agg('mean').reset_index(), 1)
result.head(3)
# -

fig = px.line(result, x='month', y='price',
              color='year', title='2008 vs 2018')
fig.show()

# 막대 그래프
result.head(3)

from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go

# +
pio.templates.default = 'plotly_white'

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=('2008', '2018'))

for i, year in enumerate([2008, 2018]):
    data = result.loc[result['year'] == year, :]
    fig.add_trace(go.Bar(x=data['month'], y=data['price'], name=str(year),
                  row=i+1, col=1))

fig.show()
# -

# ## Butterfly plot

# +
# 변수 정리

# +
fig = go.Figure()

# 2008년
fig.add_trace(go.Bar(
    x=data_2008['price'],
    y=data_2008['month'],
    name='2008년',
    marker_color='blue',
    orientation='h', # 막대그래프를 가로형으로
    text = formatted_price_2008,
    textposition='inside'
))
# 2018년
fig.add_trace(go.Bar(
    x=data_2018['price'],
    y=data_2018['month'],
    name='2018년',
    marker_color='red',
    orientation='h', # 막대그래프를 가로형으로
    text = formatted_price_2018,
    textposition='inside'
))


fig.update_layout(
    title='평균가격 비교 : 2008년 vs 2018년',
    xaxis_title = '',
    showlegend=False,
    barmode='relative',
    bargap=0.25,
    height=500,
    # 주석
    annotations=[
        dict(x=0.25, y=1.07, xref='paper', yref='paper', text='2008',
             showarrow=False, font=dict(color='blue', size=14)),
        dict(x=0.25, y=1.07, xref='paper', yref='paper', text='2008',
             showarrow=False, font=dict(color='red', size=14))
    ]
)
fig.show()
# -

# ## boxplot

pio.templates.default = ''

df = sales[sales['year'].isin([2007, 2008, 2009, 2010])]
print(df.shape)
df.head(3)

df = df[df['price'] <= 700000]
print(df.shape)
df.head(3)

fig = px.box(df, x='year', y='price', color='propertyType',
               points='all', hover_data=df.columns)
fig.show()

df.columns
[f'{col}:{df[col]}' for col in df.columns]


def format_hover_text(df):
    return df.apply(lambda row : '<br>'.join([f'{col}:{row[col]}' for col in df.columns]), axis=1)


format_hover_text(sales.loc[sales['propertyType'] == 'unit'])

# +
fig = go.Figure()

fig.add_trace(
    go.Box(
        x=sales.loc[sales['propertyType'] == 'house', 'year'],
        y=sales['price'], name='house',
        boxpoints='all', pointpos=-1.8
    )
)
fig.add_trace(
    go.Box(
        x=sales.loc[sales['propertyType'] == 'unit', 'year'],
        y=sales['price'], name='unit',
        hoverinfo='text',
        text=format_hover_text(sales.loc[sales['propertyType'] == 'unit'])
    )
)

fig.update_traces(hoverlabel=dict(bgcolor='rgba(255, 99, 71, 0.3)'),
                  selector=dict(name='unit'))
fig.update_layout(boxmode='group')
fig.show()
# -

# plotly chart
import plotly.express


