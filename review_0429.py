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

# # Matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# ## 그래프 그리기 (기본)

x = np.arange(1, 10, 1)
y = np.arange(11, 20, 1)
print(x, y)

# +
# 선그래프
plt.plot(x, y)

# 제목, 축이름 넣기
plt.title('My Plot', size=15)
# x축 이름
plt.xlabel('x')
# y축 이름
plt.ylabel('y', rotation=0)

# 순수한 그래프만 그려줘 !
plt.show()

# +
# subplot 이용하기
fig, ax = plt.subplots()

# 그래프 그리기
ax.plot(x, y)

# 제목 넣기
ax.set_title('Line plot with subplot')
# x축 이름
ax.set_xlabel('x')
# y축 이름
ax.set_ylabel('y', rotation=0)

# 다른 옵션 추가하기
# dpi(dot per inch) : 해상도
fig.set_dpi(300)

# 그래프 저장하기
fig.savefig('second_plt.png')

# 그래프 보여주기
plt.show()

# +
# subplot 이용해서 여러 그래프 그리기
x = np.linspace(0, np.pi * 4, 100)

# 3x1
fig, axes = plt.subplots(3, 1)

# sin
axes[0].plot(x, np.sin(x))
# cos
axes[1].plot(x, np.cos(x))
# tan
axes[2].plot(x, np.tan(x))

plt.show()
# -

# ## 주요 그래프

# +
# lineplot
# 그래프 바탕 만들어주기
fig, ax = plt.subplots(figsize=(4, 9))

# 변수
x = np.arange(-5, 6, 1)
y = x ** 3

# 그래프 그리기
ax.plot(x, y,
        linestyle=':',
        marker='*',
        color='darkblue')

# 그래프 보여주기
plt.show()

# +
# color
x = np.arange(-10, 11, 1)

fig, ax = plt.subplots()

# 그래프 그리기
ax.plot(x, 2*x, color='r')
ax.plot(x, 3*x, color='g')
ax.plot(x, (-1)*x, color='0.8')
ax.plot(x, (-2)*x, color='#524FA1')

# 그래프 보여주기
plt.show()

# +
# marker
x = np.arange(-5, 6, 1)

# 배경 만들기
fig, ax = plt.subplots(figsize=(4, 9))

# 그래프 그리기
ax.plot(x, x**2, marker='.')
ax.plot(x, x**2 + 1, marker='o')
ax.plot(x, (x-1)**2, marker='v')
ax.plot(x, 2*(x**2), marker='s')
ax.plot(x, (x+1)**2, marker='*')

# 그래프 보여주기
plt.show()

# +
# 축 경계 조절
x = np.linspace(0, 10, 100)

# 그래프 바탕 그리기
fig, ax = plt.subplots(figsize=(6, 8))

# 그래프 그리기
ax.plot(x, np.sin(x))
ax.plot(x, np.cos(x))
ax.plot(x, np.tan(x))

# 축 경계 조절하기
ax.set_xlim(-1, 11)
ax.set_ylim(-5, 5)

# 그래프 보여주기
plt.show()

# +
# 범례
# 그래프 배경 그리기
fig, ax = plt.subplots(figsize=(4, 9))

# 변수
x = np.linspace(-3, 4, 100)

ax.plot(x, x**2, label='y=x^2')
ax.plot(x, x**3, label='y=x^3')

# 그래프 제목, 축 이름 설정
ax.set_title('Legend of graph')
ax.set_xlabel('x')
ax.set_ylabel('y', rotation=0)

# 축 경계 설정
ax.set_xlim(-3, 3)
ax.set_ylim(-30, 30)

# 범례 설정
ax.legend(
    loc='best',
    shadow=True,
    fancybox=True,
    borderpad=1
)

# 그래프 보여주기
plt.show()
# -








