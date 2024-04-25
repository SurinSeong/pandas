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

# numpy의 array는 단일 데이티 타입
# 정수와 실수가 혼용되어 있으면 실수로 변경된다.
print(np.array([3, 1.4, 2, 3, 4.5]))

np.array([[1, 2], [4, 5]])

# +
print(np.array([1, 2, 3, 4]))

# dtype 옵션 : data type 명시
print(np.array([1, 2, 3, 4], dtype=float))

# astype : 데이터 타입을 변경
arr = np.array([1, 2, 3, 4], dtype=float)
print(arr)
print(arr.dtype)
print(arr.astype(int))
# -

# np.zeros()
np.zeros(10, dtype=int)

# np.ones()
np.ones((3, 5), dtype=float)

# np.arange(start, stop, step)
np.arange(0, 20, 2)

# np.linspace() : n분의 1
np.linspace(0, 1, 5)

# 난수 활용, 배열 만들기
print(np.random.random((2, 2)))
print(np.random.rand(2, 2))

# ### np.random.random vs. np.random.rand
# * 0 ~ 1 사이 균일 분포 난수 생성 (기능으로는 같음)
# * np.random.random : size를 튜플 형태로 매개변수로 받아서 난수 생성
# * np.random.rand : 직접 차원을 넣어줌.

# np.random.normal / np.random.randn ==> 정규분포
print(np.random.normal(0, 1, (2, 2))) # 평균, 표준편차, size(tuple 형태)
print(np.random.randn(2, 2))

# randint : 0 ~ 10-1 범위에서 정수로 이루어진 행렬 생성
np.random.randint(0, 10, (2, 2))

# 배열의 기초
x = np.random.randint(10, size=(3, 4))
print(x)
print(x.ndim)
print(x.shape)
print(x.size)
print(x.dtype)

# indexing
a = np.arange(7)
print(a)
print(a[3])
print(a[-3])

a[0] = 10
print(a)

# +
# slicing
print(a)
print(a[1:4])
print(a[1:])

# 중요중요
print(a[::2]) # interval이 2칸씩
print(a[::-1]) # 역순

# +
# reshape (형태 변경)
x = np.arange(8)
print(x)
print(x.shape)

x2 = x.reshape((2, 4))
print(x2)
print(x2.shape)
# -

# concatenate : array 이어붙이기
x1 = np.array([0, 1, 2])
x2 = np.array([3, 4, 5])
print(x1)
print(x2)
print(np.concatenate([x1, x2]))

# concatenate에서 axis 변경하기
m = np.arange(4).reshape(2, 2)
print(m)
print(np.concatenate([m, m])) # default : axis=0
print(np.concatenate([m, m], axis=1))

m = np.arange(16).reshape(4, 4)
print(m)
print(np.split(m, [3], axis=0))
print(np.split(m, [3], axis=1))

m3, m4 = np.split(m, [3], axis=1)
print(m3)
print(m4)


# ### Numpy 연산

# array의 모든 원소에 5를 더하는 함수 생성
# for 반복문 이용
def 더하기_5(values):
    output = np.empty(len(values))
    
    for i in range(len(values)):
        output[i] = values[i] + 5

    return output


nums = np.random.randint(1, 10, size=5)
print(nums)

print(더하기_5(nums))

# numpy 특성 이용
big_arr = np.random.randint(1, 10, size=10)
print(big_arr)
print(big_arr + 5)

np.empty(5) # 5개의 원소를 가진 1차원 배열 생성
# 초기화는 되어있지 않음 > 여기에 출력되는 값은 메모리에 이미 존재한 임의의 값으로 배열을 채운다.

# 기본 연산
x = np.arange(4)
print(x + 5)
print(x - 5)
print(x * 5)
print(x / 5)

# +
# 행렬 간 연산
x = np.arange(4).reshape((2, 2))
print(x)

y = np.random.randint(10, size=(2, 2))
print(y)

print(x + y)
print(x - y)
print(y / x)
print(x * y)
# -

# Broadcasting
print(x)
print(x + 5)

# 집계함수
x = np.arange(8).reshape((2, 4))
print(x)
print(np.sum(x))
print(np.mean(x))
print(np.min(x))
print(np.max(x))

# axis 사용, 집계하기
print(x)
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

# +
x = np.arange(5)

print(x < 3)
print(x > 5)
# -

# 특정 정보를 추출하고 싶을 때
x[x < 3]

# 실무에서
condition = x < 3
x[condition]
