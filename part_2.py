import numpy as np
from scipy.integrate import quad


def create_matrix(m, n, λ, μ):
    matrix = np.zeros((m + n + 1, m + n + 1))
    for i in range(m + n):
        matrix[i, i + 1] = λ
        if i < m:
            matrix[i + 1, i] = μ * (i + 1)
        else:
            matrix[i + 1, i] = μ * m
    return matrix


def probability(matrix):
    diag = np.diag([matrix[i, :].sum() for i in range(matrix.shape[0])])
    matrixT = matrix.T - diag
    matrixTM = matrixT
    matrixTM[-1, :] = 1
    B = np.zeros(matrixTM.shape[0])
    B[-1] = 1
    return np.linalg.inv(matrixTM).dot(B)

def average_length(p):
    summ = 0
    for i in range(1, n + 1):
        summ += i * p[m + i]
    return summ


def average_time(p, m, μ, n):
    summ = 0
    for i in range(0, n):
        summ += (i + 1) / (m * μ) * p[m + i]
    return summ


def busy_channels(p, m, n):
    summ1, summ2 = 0, 0
    for i in range(m + 1, m + n + 1):
        summ1 += m * p[i]
    for i in range(1, m + 1):
        summ2 += i * p[i]
    return summ1 + summ2



m = 7
n = 6
λ = 36
μ = 7


matrix = create_matrix(m, n, λ, μ)
probabil = probability(matrix)


print(f'a) Установившиеся вероятности:  {probabil}')
print(f'b) Вероятность отказа в обслуживании: {probabil[-1]}')
print(f'c) Относительная интенсивность обслуживания: {1 - probabil[-1]}')
print(f'c) Абсолютная интенсивность обслуживания: {(1 - probabil[-1]) * λ}')
print(f'd) Средняя длина очереди: {average_length(probabil)}')
print(f'e) Среднее время в очереди: {average_time(probabil, m, μ, n)}')
print(f'f) Среднее число занятых каналов: {busy_channels(probabil, m, n)}')
print(f'g) Вероятность того, что поступающая заявка не будет ждать в очереди: {sum(probabil[:m])}')
print(f'h) Среднее время простоя системы массового обслуживания: {1 / λ}')
print(f'Матрица интенсивностей: \n{matrix}')
