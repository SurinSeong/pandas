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

# # Matplotlib - plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# ## DF 불러오고 확인하기

# +
# 경로 설정
path = '../data/raw_sales.csv'

# 데이터 불러오기
raw = pd.read_csv(path, parse_dates=['datesold']) # 지정된 열을 날짜/시간 형식으로 자동 변환해주는 옵션을 넣어주었음.
# -

raw.info() # 변환된 'datesold'는 datetime 형식으로 변환된다는 것을 확인함.

# 불러온 원본 데이터는 복사해서 원본 유지하면서 사용하는 것이 좋음.
sales = raw.copy()

# DF 확인용 과정 - head(), shape(), info()
print(sales.shape)
sales.head(1)

sales.info()

# 'datesold'에서 년도 추출해서 따로 저장하기
sales['year'] = sales['datesold'].dt.year

# +
# # 'year' 컬럼 잘 들어간 것 확인하기
# sales.head(1)
# -

# 년도별 평균 가격 확인하기 - round()로 반올림값 넣기, 변수에 저장하기
result = round(sales.groupby('year')['price'].agg('mean'), 1)

# 'result'의 정보 확인
result[:5]

print(type(result)) # Series
print(result.index, result.values)

# ## Graph

# +
# 시간의 변동에 따른 추세 확인하기
# 선 그래프 (line graph)

# 그래프 바탕 설정
fig, ax = plt.subplots(figsize=(12, 6))

# 그래프 그리기
ax.plot(result.index, result.values) # plot(x, y)

# 그래프 꾸미기
ax.set_title('Average of Price per Year', size=15)
ax.set_xlabel('year', size=10)
ax.set_ylabel('price', size=10)

# 그래프 저장하기
# plt.savefig('../data/img/average_of_price_per_year.png')
plt.show()
# -

# 'month' 컬럼 만들기
sales['month'] = sales['datesold'].dt.month

# 잘 들어갔는지 확인
sales.head(1)

# +
# 년도가 2007 ~ 2009인 데이터만 찾기
condition = sales['year'].isin([2007, 2008, 2009])

# df에 조건을 설정한 DF를 변수로 저장하기
df = sales.loc[condition, :]
# -

# df 확인
df.tail(1)

# 연도별, 월별 평균 가격을 확인하고 변수에 저장하기
result = round(df.groupby(['year', 'month'])['price'].agg('mean'), 1)

# 저장한 내용
result.head()

print(type(result))
result.index

# reset_index를 이용해서 MultiIndex를 풀어 DF로 만들고 저장하기
result = result.reset_index()

result.iloc[::3, :]

# 년도 데이터의 값 가져오기 (겹치지 않게)
result['year'].unique()

# loc 이용해서 'year'가 2007인 값 가져오기, 년도별로 저장하기
result_2007 = result.loc[result['year'] == 2007, :]
result_2008 = result.loc[result['year'] == 2008, :]
result_2009 = result.loc[result['year'] == 2009, :]

# DF 저장한 것 확인하기
print(result_2007[:3])
print(result_2008[:3])
print(result_2009[:3])

# +
# 여러 그래프 그리기 - subplot 이용
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))

# bar(x, y)
# 첫 번째 subplot : 0부터 시작
ax[0].bar(result_2009['month'], result_2009['price'])
ax[0].set_title('year_2009')

# 두 번째 subplot
ax[1].bar(result_2008['month'], result_2008['price'])
ax[1].set_title('year_2008')

# 세 번째 subplot
ax[2].bar(result_2007['month'], result_2007['price'])
ax[1].set_title('year_2007')

# 그래프끼리 겹치는 것 막아주기
plt.tight_layout()
plt.show()
# -

print(len(result['year'].unique()))
type(result['year'].unique())

# +
N = len(result['year'].unique()) # 2007, 2008, 2009

# 바탕 만들기
fig, ax = plt.subplots(nrows=N, ncols=1, figsize=(6, 3*N))

# for문 돌려서 그래프 그리기
for i, year in enumerate(result['year'].unique()):
    # print(i, year)
    
    # 그래프용 데이터 설정
    data = result.loc[result['year'] == year]

    # 그래프 그리기
    ax[i].bar(data['month'], data['price'])
    ax[i].set_title(f'year{year}')

# 겹침 없애기
plt.tight_layout()
plt.show()

# +
# 배경 만들기
fig, ax = plt.subplots(figsize=(6, 3))

# 겹쳐 그리기
ax.bar(result_2009['month'], result_2009['price'])
ax.bar(result_2008['month'], result_2008['price'])
ax.bar(result_2007['month'], result_2007['price'])

plt.show()
# -

# # Seaborn

# +
# 년도별 집값 평균 시각화
# 배경 설정하기
fig, ax = plt.subplots(figsize=(10, 6))

# 그래프 그리기
sns.lineplot(data=sales, x='month', y='price', ax=ax)

# 그래프 꾸미기
ax.set_title('Average of House price per year with seaborn', size=15)
ax.set_xlabel('Year', size=10)
ax.set_ylabel('Price', size=14, labelpad=12)

plt.show()
# -

# sales 확인
print(sales.head(1))
print(sales.tail(1))

# sales의 년도 확인
sales['year'].unique()

# 조건 (2007 ~ 2010) 적용해서 DF 만들기
condition = sales['year'].isin([2007, 2008, 2009, 2010])
df = sales.loc[condition, :]

print(df.head(1))
print(df.tail(1))

# +
# df로 그래프 만들기
# 배경 설정
fig, ax = plt.subplots(figsize=(10, 6))

# boxplot 만들기
sns.boxplot(data=df, x='year', y='price', hue='propertyType', ax=ax)

# 그래프 보여주기
plt.show()
# -

# ## boxplot

# +
data = {'values' : [10, 11, 13, 14, 15, 16, 18, 20, 100, 120, -1]}

# DF 만들기
temp_df = pd.DataFrame(data)
temp_df[:3]

# +
# 1단계 : 사분위수 확인하기 (Q1, Q2 : median, Q3, IQR)
Q1 = temp_df['values'].quantile(0.25)
Q3 = temp_df['values'].quantile(0.75)
IQR = Q3 - Q1

print(Q1, Q3, IQR)

# +
# 2단계 : 이상치 임계값 (threshold) 설정하기
lower_bound = Q1 - 1.5 * IQR # 하한가
upper_bound = Q3 + 1.5 * IQR # 상한가

print(lower_bound, upper_bound)
# -

# 3단계 : 이상치 확인하고 변수에 이상치 DF를 저장하기
condition = (temp_df['values'] < lower_bound) | (temp_df['values'] > upper_bound)
outlier = temp_df.loc[condition]

# 이상치 제거한 것 추출하기
temp_df.loc[~condition]

# 4단계 : 이상치 처리 (a. 제거, b. upper_bound & lower_bound로 바꾸기)
# a. 이상치 제거 >> 함수 : remove_outlier
df_clean = temp_df.loc[~condition]

# +
# b. upper_bound & lower_bound로 바꾸기
# 원본 훼손하지 않기 위해 복사해서 사용함.
df_change = temp_df.copy()

# 이상치 처리하기
df_change['values'] = df_change['values'].apply(lambda x : min(x, upper_bound))
df_change['values'] = df_change['values'].apply(lambda x : max(x, lower_bound))
df_change


# -

# 1 ~ 4단계를 합쳐서 이상치 처리 함수 만들기
# 이상치 제거 함수
# 필요한 변수 : DF, 변경할 column명
def remove_outliers(df, column):

    # 사분위수 설정
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # 임계값 설정
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치의 조건
    condition = (df[column] < lower_bound) | (df[column] > upper_bound)

    # 이상치 DF
    outliers = df.loc[condition]
    # 이상치 제거 DF
    df_clean = df.loc[~condition].reset_index(drop=True)

    # 이상치 제거한 DF와 이상치 DF 확인 가능
    return df_clean, outliers
    


# 실제 데이터에 적용해보기
df_clean, outliers = remove_outliers(df, 'price')
print(df_clean, '\n', outliers)

# 과학적 표기법 비활성 설정
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter()
formatter.set_scientific(False)

# +
# 2007 ~ 2010 추출 결과 데이터 시각화 - 이상치 제거 전, 후
# 배경 설정
fig, ax = plt.subplots(nrows=2, figsize=(10, 12))

# 첫 번째 boxplot 그리기
sns.boxplot(data=df, x='year', y='price', hue='propertyType', ax=ax[0])
# 꾸미기
ax[0].set_title('with Outliers')
ax[0].yaxis.set_major_formatter(formatter)

# 두 번째 boxplot 그리기
sns.boxplot(data=df_clean, x='year', y='price', hue='propertyType', ax=ax[1])
# 꾸미기
ax[1].set_title('without Outliers')

# 겹치지 않게
plt.tight_layout()
plt.show()
# -

# ## tips dataset

# seaborn의 데이터 가져오기
tips = sns.load_dataset('tips')

# +
# tips 관련 그래프 그리기
# 요일별 총 청구금액 시각화

# 배경 설정 : 2x2
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

# (0, 0) 그리기
sns.barplot(data=tips, x='day', y='total_bill', ax=ax[0, 0])
# 제목
ax[0, 0].set_title('Total Bill per day')

# (0, 1) 그리기
sns.countplot(data=tips, x='day', ax=ax[0, 1])
ax[0, 1].set_title('Tips per day')

# (1, 0) 그리기
sns.stripplot(data=tips, x='day', y='total_bill', ax=ax[1, 0])
ax[1, 0].set_title('Stripplot of Tips')

# (1, 1) 그리기
sns.pointplot(data=tips, x='day', y='total_bill', ax=ax[1, 1])
ax[1, 1].set_title('Pointplot of Tips')

# 겹침 없앰
plt.tight_layout()
plt.show()

# +
# 2x2 서브플롯 만들기 (프롯 안에 인덱스 넣기)
# 행과 열을 설정
nrows, ncols = 2, 2

# 배경 만들기
fig, ax = plt.subplots(nrows, ncols, figsize=(6, 6))

# 각 서브플롯 안에 인덱스 표시하기
for i in range(nrows):
    for j in range(ncols):
        # 글자 넣기
        ax[i, j].text(0.5, 0.5, f'INDEX {i}, {j}', ha='center', va='center', fontsize=12)
        # x축 눈금 설정
        ax[i, j].set_xticks([])
        # y축 눈금 설정
        ax[i, j].set_yticks([])

plt.tight_layout()
# # 이미지 저장
# plt.savefig('../data/img/subplots.png')
plt.show()
# -

# # ploty

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default = 'jupyterlab'

# +
# boxplot 그리기
# 데이터 불러오기
df = px.data.tips()

# 그래프 그리기
fig = px.box(df, y='total_bill')

# 그래프 보여주기
fig.show()

# +
# 가상의 데이터 만들기
x = np.arange(0, 20, 1)
y1, y2 = x**2, x**3
# print(y1, y2)

# figure 형성
fig = go.Figure()

# 시각화 차트
# 지정한 x, y 값을 plotly를 이용해 산점도를 그래프 트레이스에 더해주기
# fig.add_trace(go.Scatter(x=x, y=y1))
# fig.add_trace(go.Scatter(x=x, y=y2))

# for문 이용해서 그리기
for y in [y1, y2]:
    fig.add_trace(go.Scatter(x=x, y=y))

# 시각화 자료 꾸미기
fig.update_layout(title='basic graph', xaxis_title='x축', yaxis_title='y축', template='plotly_white')

# 그래프 보여주기
fig.show()

# +
data = {'x' : ['a', 'b', 'c', 'd'],
        'y1' : [20, 25, 23, 27],
        'y2' : [6, 18, 3, 10]}

# 기본 차트
fig = px.bar(data, x='x', y='y1', title='basic chart')

# 차트 추가
fig.add_trace(go.Scatter(x=data['x'], y=data['y2'],
                         mode='lines+markers', name='line chart'))

# 막대 차트
fig.update_traces(marker_color='lightblue', selector=dict(type='scatter')) # selecter : type이 scatter(산점도) 타입의 trace만 적용

# 그래프 보여주기
fig.show()
# -

# ## 선 그래프 (Line graph)

# sales 사용
# 연도별 가격 평균
result = sales.groupby('year')['price'].agg('mean')
result[:5]

# 그래프 그리기
fig = px.line(result, x=result.index, y=result.values,
              title='연도별 평균 집 값 추이')
fig.show()

# sales 컬럼 확인
sales.columns

# 날짜에서 달을 추출하기
sales['month'] = sales['datesold'].dt.month
sales['year'] = sales['datesold'].dt.year

sales.head(2)

sales.tail(2)

# 2008년과 2018년 비교하기
condition = sales['year'].isin([2008, 2018])
df = sales[condition]

df.head(1)

result = df.groupby(['year', 'month'])['price'].agg('mean').reset_index()

result.head(1)

# +
# 그래프 그리기
pio.templates.default = 'plotly_white'

fig = px.line(result, x='month', y='price', color='year', title='2008 vs 2018 월별 집값 평균 비교')
fig.show()

# +
# 다중 차트 그리기
pio.templates.default = 'plotly_white'

fig = make_subplots(rows=2, cols=1, subplot_titles=('2008년 차트', '2018년 차트'))

for i, year in enumerate([2008, 2018]):
    # 데이터 설정
    data = result.loc[result['year'] == year, :]
    # 그래프 그리기
    fig.add_trace(go.Bar(x=data['month'], y=data['price'], name=str(year)), row=i+1, col=1)

# 그래프 꾸미기
fig.update_layout(title='Average House Price by Month and Year',
                  xaxis_title='month',
                  yaxis_title='price',
                  height=500)
# 그래프 보여주기
fig.show()

# +
# 새로운 컬럼 추가하기
# map
month_map = {1 : '1월', 2 : '2월', 3 : '3월', 4 : '4월', 5 : '5월', 6 : '6월',
             7 : 'JULY', 8 : 'AUG', 9 : 'SEP', 10 : 'OCT', 11 : 'NOV', 12 : 'DEC'}

# map한 '월' 생성
result['month_map'] = result['month'].map(month_map)
# -

result.head(1)

# +
# 막대그래프 만들기
fig = go.Figure()

# 막대 그래프 그리기
for year in [2008, 2018]:
    # 데이터 설정
    yearly_data = result[result['year'] == year]

    fig.add_trace(go.Bar(x=yearly_data['month_map'],
                         y=yearly_data['price'],
                         name=str(year),
                         marker_color='blue' if year == 2008 else 'red'))
# 그래프 레이아웃 설정
fig.update_layout(barmode='group',
                  title='평균 가격 비교 (2008 vs 2018)',
                  xaxis=dict(tickmode='array',
                             tickvals=list(month_map.values()),
                             ticktext=list(month_map.values())),
                  yaxis_title='평균가격',
                  height=500)
# 그래프 보여주기
fig.show()
# -

# ### Butterfly Chart 만들기

data_2008 = result.loc[result['year'] == 2008, :]
data_2018 = result.loc[result['year'] == 2018, :]

# 천 단위 구분하기
formatted_price_2008 = [f'{x:,.0f}' for x in data_2008['price']]
formatted_price_2018 = [f'{x:,.0f}' for x in data_2018['price']]

print(formatted_price_2008, formatted_price_2018)

# +
# 그래프 만들기
fig = go.Figure()

# 2008년
fig.add_trace(go.Bar(x=data_2008['price'],
                     y=data_2008['month'],
                     name='2008년',
                     marker_color='blue',
                     orientation='h', # 막대 그래프를 가로형으로
                     text=formatted_price_2008,
                     textposition='inside'))

# 2018년
fig.add_trace(go.Bar(x=data_2018['price'],
                     y=data_2018['month'],
                     name='2018년',
                     marker_color='red',
                     orientation='h', # 막대 그래프를 가로형으로
                     text=formatted_price_2018,
                     textposition='inside'))

# 레이아웃 업데이트
fig.update_layout(title='평균가격 비교 : 2008년 vs 2018년',
                  xaxis_title='',
                  showlegend=False,
                  barmode='relative',
                  bargap=0.25,
                  height=500,
                  annotations=[dict(x=0.25, y=1.07, xref='paper', yref='paper', text='2008', showarrow=False, font=dict(color='blue', size=14)),
                               dict(x=0.25, y=1.07, xref='paper', yref='paper', text='2018', showarrow=False, font=dict(color='red', size=14))])
# 그래프 보여주기
fig.show()
# -

# ### boxplot

df = sales[sales['year'].isin([2007, 2008, 2009, 2010])]

df.head(1)

df = df[df['price'] <= 700000]

# boxplot 그리기
fig = px.box(df, x='year', y='price', color='propertyType', points='all', hover_data=df.columns)
fig.show()

[f'{col}:{df[col]}' for col in df.columns]


# 함수
def format_hover_text(df):
    return df.apply(lambda row : '<br>'.join([f'{col}:{df[col]}' for col in df.columns]), axis=1)


format_hover_text(sales.loc[sales['propertyType'] == 'unit'])

# +
# 그래프 만들기
fig = go.Figure()

# 그래프 만들기 1
fig.add_trace(
    go.Box(
        x=sales.loc[sales['propertyType'] == 'house', 'year'],
        y=sales['price'], name='house',
        boxpoints='all',
        pointpos=(-1.8)
    )
)
# 그래프 만들기 2
fig.add_trace(
    go.Box(
        x=sales.loc[sales['propertyType'] == 'unit', 'year'],
        y=sales['price'],
        name='unit',
        hoverinfo='text',
        text=format_hover_text(sales.loc[sales['propertyType'] == 'unit'])
    )
)

# trace 업데이트
fig.update_traces(
    hoverlabel=dict(bgcolor='rgba(255, 99, 71, 0.3)'),
    selector=dict(name='unit')
)

# 레이아웃 업데이트
fig.update_layout(boxmode='group')

# 그래프 보여주기
fig.show()

# +
# 그래프 만들기
fig = go.Figure()

# 그래프 만들기 1
fig.add_trace(
    go.Box(
        x=df.loc[df['propertyType'] == 'house', 'year'],
        y=df['price'],
        name='house',
        boxpoints='all',
        pointpos=(-1.8)
    )
)
# 그래프 만들기 2
fig.add_trace(
    go.Box(
        x=df.loc[df['propertyType'] == 'unit', 'year'],
        y=df['price'],
        name='unit',
        hoverinfo='text',
        text=format_hover_text(df.loc[df['propertyType'] == 'unit'])
    )
)

# trace 업데이트
fig.update_traces(
    hoverlabel=dict(bgcolor='rgba(255, 99, 71, 0.3)'),
    selector=dict(name='unit')
)

# 레이아웃 업데이트
fig.update_layout(boxmode='group')

# 그래프 보여주기
fig.show()
# -

# ### plotly chart

df = px.data.tips()

df.head(1)

avg_df = df.groupby('day')[['total_bill', 'tip']].mean().reset_index()

day_order = ['Thursday', 'Friday', 'Saturday', 'Sunday']
day_full_names = {'Thur':'Thursday', 'Fri':'Friday', 'Sat':'Saturday', 'Sun':'Sunday'}

avg_df['day'] = avg_df['day'].map(day_full_names)

avg_df

avg_df['day'] = pd.Categorical(avg_df['day'], categories=day_order)

avg_df

avg_df.info()

avg_df = avg_df.sort_values('day').reset_index(drop=True)

avg_df['day'].values

# +
# 그래프 만들기
# 막대 그리프 그리기
fig = px.bar(
        avg_df,
        x='day',
        y='tip',
        labels={'tip' : 'Average Total Bill'},
        text_auto='.2s',
        color='day',
        color_discrete_map={'Thursday' : 'gray', 'Friday' : 'gray', 'Saturday' : 'gray', 'Sunday' : 'red'}
        )
# trace 업데이트
fig.update_traces(textfont_size=12, textangle=0, textposition='outside', cliponaxis=False)

# 색 지정
colors = ['red' if day == 'Sunday' else 'gray' for day in avg_df['day'].values]

# 그래프 추가
fig.add_trace(
    go.Scatter(
        x=avg_df['day'],
        y=avg_df['total_bill'],
        mode='lines+markers',
        name='average_bill',
        line=dict(color='darkgray', width=2),
        marker=dict(color=colors, size=8)
    )
)

# y축 정의 grid lines
max_value = max(avg_df['total_bill'].max(), avg_df['tip'].max())
yaxis_range = round(max_value + 7 - (max_value % 10))

# x축, y축 범례 정의하기
fig.update_layout(
    title='Average Total Bill and Tip by Day',
    xaxis=dict(
        title='Day of the Week',
        tickmode='array',
        tickvals=day_order
    ),
    yaxis=dict(
        title='Average Amount (US $)',
        range=[0, yaxis_range],
        tickmode='linear',
        tick0=0,
        dtick=5,
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.5,
        griddash='dot'
    ),
    plot_bgcolor='white',
    legend_title='Data Type',
    legend=dict(
        orientation='h',
        x=0.5,
        xanchor='center',
        y=(-0.5),
        yanchor='bottom',
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='Black'
    )
)

# y축 grid line 추가
for i in range(10, yaxis_range + 1, 10):
    fig.add_shape(
        type='line',
        x0=(-0.5),
        y0=i,
        x1=3.5,
        y1=i,
        line_dash='dot',
        line=dict(color='black', width=1)
    )

fig.show()
# -

max_value = max(avg_df['total_bill'].max(), avg_df['tip'].max())
print(max_value)

yaxis_range = round(max_value + 7 - (max_value % 10))
print(yaxis_range)




