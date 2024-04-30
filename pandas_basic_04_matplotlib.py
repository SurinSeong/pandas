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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# +
# 그래프 그려보기
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

plt.plot(x, y)
plt.show() # 순수한 그래프만 그려줌

# +
# 제목, 라벨 만들기
# 위의 x, y 이용

plt.plot(x, y)

# 제목
plt.title('My first Plot')
# x축명
plt.xlabel('x')
# y축명
plt.ylabel('y')
plt.show() # 순수한 그래프만 그려줌

# +
# subplots() : 하나의 큰 바탕에 여러 그래프 그리기
# 위의 x, y 이용
fig, ax = plt.subplots()

ax.plot(x, y)
ax.set_title('My first Plot')
ax.set_xlabel('x')
ax.set_ylabel('y', rotation=0)

# 그래프에 옵션 넣기
# dpi : 해상도 (dot per inch)
fig.set_dpi(300)
# 그래프 저장하기
fig.savefig('first_plt.png')

plt.show()

# +
# 여러 개 그래프 그리기

x = np.linspace(0, np.pi*4, 100)

# 2x1
fig, axes = plt.subplots(2, 1)
# sin 그래프
axes[0].plot(x, np.sin(x))
# cos 그래프
axes[1].plot(x, np.cos(x))

plt.show()
# -

# ## matplotlib 주요 그래프

# +
# 선 그래프 (line plot)
fig, ax = plt.subplots()

x = np.arange(15)
y = x ** 2

ax.plot(x, y,
        linestyle=':',
        marker='*',
        color='darkblue')
plt.show()
# -

# color
x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, color='r') # 'r' : 빨간색
ax.plot(x, x + 2, color='g')
ax.plot(x, x + 4, color='0.8') # 0.8 : gray scale (0 : 검정, 1 : 하양)
ax.plot(x, x + 6, color='#524FA1')
plt.show()

# +
# marker
x = np.arange(10)

fig, ax = plt.subplots()

ax.plot(x, x, marker='.')
ax.plot(x, x+2, marker='o')
ax.plot(x, x+4, marker='v')
ax.plot(x, x+6, marker='s') # s : square
ax.plot(x, x+8, marker='*')

plt.show()

# +
# 축 경계 조정하기

# 0~10까지에서 100개로 나누기
x = np.linspace(0, 10, 100)

fig, ax = plt.subplots()

ax.plot(x, np.sin(x))
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 2)

plt.show()

# +
# 범례(legend)
fig, ax = plt.subplots()

ax.plot(x, x, label='y=x')
ax.plot(x, x ** 2, label='y=x^2')

# 그래프 제목, 축 이름 설정
ax.set_title('legend of graph')
ax.set_xlabel('x')
ax.set_ylabel('y', rotation=0)

# 범례 설정
ax.legend(loc='best',
          shadow=True,
          fancybox=True,
          # borderpad=2
         )

plt.show()

# +
# scatter plot (산점도) : 꼭 그려봐야 함.
x = np.arange(10)

fig, ax = plt.subplots()
ax.scatter(
    x, x**2
    # markersize=15,
    # markerfacecolor='white',
    # markeredgecolor='blue'
)

plt.show()

# +
# bubble 형태, scatter
x = np.random.randn(50)
y = np.random.randn(50)
colors = np.random.randint(0, 100, 50)
sizes = 500 * np.pi * np.random.rand(50) ** 2

fig, ax = plt.subplots()
# alpha : 투명도
ax.scatter(x, y, c=colors, s=sizes, alpha=0.3)

plt.show()

# +
# 바 그래프 (bar plot)
# bar
x = np.arange(10)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x, x*2)

plt.show()

# +
# 누적 바 그래프 그리기

x = np.random.rand(3)
y = np.random.rand(3)
z = np.random.rand(3)

data = [x, y, z]

fig, ax = plt.subplots()

# 이 코드는 중요함 !
x_ax = np.arange(3)
for i in x_ax:
    ax.bar(x_ax,
           data[i],
           bottom=np.sum(data[:i], axis=0)) # 세로방향

ax.set_xticks(x_ax)
ax.set_xticklabels(['A', 'B', 'C'])

plt.show()

# +
# 히스토그램
data = np.random.randn(1000)

fig, ax = plt.subplots()
ax.hist(data, bins=50)

plt.show()
# -
# ## Matplotlib with pandas

# 상대경로로 찾기
path = '../data/president_heights.csv'

# 변수에 담기 전에 열어보고 담기
pd.read_csv(path).head(3)

pd.read_csv(path).reset_index(drop=True).head(3)

df_ph = pd.read_csv(path).reset_index(drop=True)
df_ph.head(3)

# +
# 그래프 그리기
fig, ax = plt.subplots()

ax.plot(df_ph['order'], df_ph['height(cm)'],
        label='height')

ax.set_title('President\'s Height')
ax.set_xlabel('order')
ax.set_ylabel('height(cm)')

plt.show()

# +
# 히스토그램으로 그려도 될 듯 !

# +
path2 = '../data/pokemon.csv'

df_po = pd.read_csv(path2)
df_po.head(3)
# -


df_po.columns

# +
# 조건 설정
condition_01 = (df_po['Type 1'] == 'Fire') | (df_po['Type 2'] == 'Fire')

df_po[condition_01].head(3)
# -

# 조건으로 묶인 df 변수 저장하기
fire = df_po[condition_01]
fire.head(3)

# 조건 설정
condition_02 = (df_po['Type 1'] == 'Water') | (df_po['Type 2'] == 'Water')
df_po[condition_02].head(3)

# 변수 저장
water = df_po[condition_02]
water.head(3)

# +
# 그래프
fig, ax = plt.subplots()

ax.scatter(fire['Attack'], fire['Defense'],
           color='r', label='Fire', marker='*', s=50)
ax.scatter(water['Attack'], water['Defense'],
           color='b', label='Water', marker='o', s=30)

# 제목, 축 이름 설정
ax.set_title('Fire and Water')
ax.set_xlabel('Attack')
ax.set_ylabel('Defense')

# 범례 설정
ax.legend(loc='best')

plt.show()
# -




